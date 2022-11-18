import random
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from hypro_tpp.model_runner import ModelRunner
from hypro_tpp.utils import create_folder, distance_between_event_seq

# ref: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py

class SeqGenerator(ModelRunner):

    def __init__(self, config):
        super(SeqGenerator, self).__init__(config)
        self.gen_config = config.get('sample_generation', None)

        if self.model_config['pretrained_model_dir']:
            self.model.load_state_dict(torch.load(self.model_config['pretrained_model_dir']))

        self.num_samples = self.gen_config['thinning_params']['num_samples']
        self.num_seqs = self.gen_config['thinning_params']['num_seqs']
        self.num_exp = self.gen_config['thinning_params']['num_exp']
        self.patience = self.gen_config['thinning_params']['patience']
        self.std_method = self.gen_config['std_method']
        self.look_ahead_time = self.gen_config['thinning_params']['look_ahead_time']
        self.event_num = self.model_config['num_event_types_no_pad']

        self.gen_noise = self.gen_config['gen_noise_sample']
        if self.gen_noise:
            self.gen_start_pos = self.gen_config['start_pos']
            self.num_samples_per_seq = self.gen_config['num_samples_per_seq']
            self.gen_noise_sample_len = self.gen_config['gen_noise_sample_len']
            self.max_distance = self.gen_config['gen_noise_max_distance']
            self.max_len = self.gen_config['gen_max_len']
            self.del_cost = self.gen_config['distance_del_cost']
            # load data
            self.real_train_loader, self.real_dev_loader, self.real_test_loader = self.get_dataloader(skip_padding=True)
        else:
            self.min_len = self.gen_config['min_len']
            self.max_len = self.gen_config['max_len']

        self.create_thinningsampler(self.num_samples, self.num_exp)
        self.output_dir = create_folder(self.data_config['saved_dir'])


    def truncate_seq(self, batch):
        """ Truncate the sequence, extract start events as BOS for generation  """
        time_seq, _, event_seq, batch_pad_mask, _, _ = batch

        # [batch_size]
        seq_len = np.sum(batch_pad_mask.cpu().numpy(), axis=1)
        if self.gen_noise_sample_len == 0:
            truncated_len = np.array([int(length * self.gen_start_pos) for length in seq_len])
        else:
            truncated_len = seq_len - self.gen_noise_sample_len
        sample_len = seq_len - truncated_len

        # ground truths for sampling without mask
        ground_truth_col_index = [np.arange(a, b).tolist() for a, b in zip(truncated_len, seq_len)]
        ground_truth_time_seq = [time_seq[id][col_idx].cpu().numpy() for id, col_idx in
                                 enumerate(ground_truth_col_index)]
        ground_truth_event_seq = [event_seq[id][col_idx].cpu().numpy() for id, col_idx in
                                  enumerate(ground_truth_col_index)]

        # ground_truth_tuple = (ground_truth_time_seq, ground_truth_event_seq)
        ground_truth_tuple = [(a, b) for a, b in zip(ground_truth_time_seq, ground_truth_event_seq)]

        return sample_len, truncated_len, ground_truth_tuple

    def compute_distance(self, ref_seq, sample_res, sample_len):
        # tuples of (time_seq, type_seq)
        sample_seqs = [(a[0][-sample_len:], a[1][-sample_len:]) for a in sample_res]

        # take average of all distance with different del_cost
        sample_distance = [np.mean(distance_between_event_seq(ref_seq,
                                                              sample_seq,
                                                              del_cost=self.del_cost,
                                                              trans_cost=1.0,
                                                              num_types=self.event_num)[0][0])
                           for sample_seq in sample_seqs]

        # normalized distance for modeling purpose
        if self.max_distance > 0:
            sample_distance = (np.array(sample_distance) / self.max_distance).tolist()
        else:
            if self.std_method == 'min_max':
                sample_distance = self.distance_min_max_scaler(sample_distance)
            else:
                sample_distance = self.distance_std_normalizer(sample_distance)

        return sample_distance

    def min_max_scaler(self, val, min_val, max_val):
        scaled_val = (val - min_val) / (max_val - min_val)
        return scaled_val

    def std_scaler(self, val, mean, std):
        return (val - mean) / (std + 1e-8)

    def distance_min_max_scaler(self, sample_distance):
        min_val = np.min(sample_distance) - 1e-5
        max_val = np.max(sample_distance) + 1e-5

        res = [self.min_max_scaler(x, min_val, max_val) for x in sample_distance]

        return res

    def distance_std_normalizer(self, sample_distance, softmax=True):
        mean_val = np.mean(sample_distance)
        std = np.std(sample_distance)
        res = [self.std_scaler(x, mean_val, std) for x in sample_distance]
        if softmax:
            res = [np.exp(x) for x in res]
            res = [x / sum(res) for x in res]
        return res

    def make_sample_dict(self,
                         time_seq,
                         time_delta_seq,
                         event_seq,
                         sample_res,
                         distance):
        res = dict()
        res['positive'] = [
            {
                "time_since_start": time_seq.cpu().numpy()[x],
                "type_event": event_seq.cpu().numpy()[x],
                "time_since_last_event": time_delta_seq.cpu().numpy()[x]
            } for x in range(len(time_seq))
        ]

        res['negative'] = [(sample_res[i][0],  # time
                            sample_res[i][1],  # event
                            sample_res[i][2],  # dtime
                            distance[i]
                            ) for i in range(self.num_samples_per_seq)]

        return res

    def run_noise_generation(self, real_data_loader):
        """ Noise sample generation """
        results = []
        seq_id = 0
        for batch in tqdm(real_data_loader, mininterval=2, desc=f'   - (noise sample generation) -    ',
                          leave=False):
            new_batch = [x.to(self.device) for x in batch]
            time_seq, time_delta_seq, event_seq, _, _, _ = new_batch
            # truncate seqs
            # we assume the sample_len equals to the ground truth of seq length
            sample_len, truncated_len, ground_truth_tuple = self.truncate_seq(new_batch)
            # thinning can only run in single instance mode, not in batch mode
            # for each seqs, we make some noise seqs
            for i in range(len(time_seq)):
                seq_len_i = truncated_len[i] + sample_len[i]
                if truncated_len[i] == 0 or sample_len[i] == 0 or seq_len_i > self.max_len:
                    continue
                
                seq_id += 1

                if seq_id % 100 == 0:
                    print(f'{seq_id} seqs generation done.')
                # sample event seq (include the prefix event sequence)
                sample_res = self.run_generation_multi_step(model=self.model,
                                                            sample_len=sample_len[i],
                                                            start_event=event_seq[i][
                                                                        :truncated_len[i]].cpu().numpy().tolist(),
                                                            start_time=time_seq[i][
                                                                       :truncated_len[i]].cpu().numpy().tolist(),
                                                            start_dtime=time_delta_seq[i][
                                                                        :truncated_len[i]].cpu().numpy().tolist(),
                                                            event_num=self.event_num,
                                                            patience=self.patience,
                                                            num_sample_seqs=self.num_samples_per_seq,
                                                            save_in_pkl=False)

                # compute distance between sample event seq with ground truths
                distance = self.compute_distance(ref_seq=ground_truth_tuple[i],
                                                 sample_res=sample_res,
                                                 sample_len=sample_len[i])

                # merge the real, generated sequences and distances into one dict
                merge_dict = self.make_sample_dict(time_seq[i][:seq_len_i],
                                                   time_delta_seq[i][:seq_len_i],
                                                   event_seq[i][:seq_len_i],
                                                   sample_res,
                                                   distance)

                results.append(merge_dict)

        return results

    def run_generation(self, num_seqs, min_len, max_len, max_time=10, start_type=None, start_time=0):
        # self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        thinning_sampler = self.thinning_sampler
        bos_sets = list(range(self.event_num))
        # for _ in tqdm(range(num_seqs), desc=f"   (Generate)    ", leave=False, mininterval=2):
        for _ in range(num_seqs):
            # thinning can only run in single instance mode, not in batch mode
            length = random.randint(min_len, max_len)
            types = [random.sample(bos_sets, 1)[0], ] if start_type is None else [start_type, ]
            _start_time = start_time
            times = [_start_time, ]
            dtimes = [_start_time, ]
            intens = [[0] * self.event_num]
            # for j in range(1, length - 1):
            for _ in range(0, length):
                _time_seq, _event_seq = torch.FloatTensor(times), torch.LongTensor(types)
                time_last_event = _time_seq[-1].item()
                boundary = time_last_event + max_time
                _event_prefix, _time_prefix = _event_seq.unsqueeze(0).to(self.device), _time_seq.unsqueeze(0).to(
                    self.device)
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     time_last_event, boundary, self.model, self.patience],
                    mode="unbiased"
                )
                _time_uncond = torch.sum(accepted_times * weights)
                time_uncond = float(_time_uncond)
                dtime_uncond = time_uncond - time_last_event
                times.append(time_uncond)
                dtimes.append(dtime_uncond)
                intensities_at_times = self.model.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    _time_uncond.reshape(1, 1)
                )[0, 0]
                intens.append(intensities_at_times.tolist())
                next_event_name = torch.multinomial(intensities_at_times, 1)[0].item()
                types.append(next_event_name)
            rst = [
                {
                    "time_since_start": times[x],
                    "type_event": types[x],
                    "time_since_last_event": dtimes[x],
                    "intensities": intens[x]
                } for x in range(len(times))
            ]
            results.append(rst)
            seq_id += 1
        return results

    def run(self):
        splits = ["train", "dev", "test"]
        if self.gen_config.get('gen_noise_sample', None) == True:
            for _split, data_loader in zip(splits,
                                           [self.real_train_loader, self.real_dev_loader, self.real_test_loader]):
                save_path = os.path.join(self.output_dir, f"{_split}.pkl")
                with open(save_path, "wb") as f_out:
                    data = self.run_noise_generation(data_loader)
                    pickle.dump(
                        {
                            "dim_process": self.event_num,
                            _split: data
                        }, f_out
                    )
        else:
            num_seqs = self.num_seqs
            train_num = int(num_seqs)
            dev_num = int(train_num * 0.3)
            test_num = int(train_num * 0.3)
            total_tokens = 0
            for _split, _num_seqs in zip(splits, [train_num, dev_num, test_num]):
                save_path = os.path.join(self.output_dir, f"{_split}.pkl")
                with open(save_path, "wb") as f_out:
                    data = self.run_generation(_num_seqs,
                                               self.min_len,
                                               self.max_len,
                                               self.look_ahead_time)
                    pickle.dump(
                        {
                            "dim_process": self.event_num,
                            _split: data
                        }, f_out
                    )
                    total_tokens += sum([len(x) - 1 for x in data])
            print("impatient rate: {}".format(self.thinning_sampler.patience_counter / total_tokens))
