import pickle
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from hypro_tpp.preprocess import TPPDataset, create_dataloader
from hypro_tpp.models import BaseModel
from hypro_tpp.lib import EventSampler
from hypro_tpp.utils import LogWriter, make_config_string, create_folder, save_config


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.train_config = config.get('train', None)
        self.model_config = config.get('model', None)
        self.data_config = config.get('data', None)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.update_config()
        # dont load model here when doing joint evaluation
        if not self.config.get('skip_load_model_in_base_runner', False):
            self.model = BaseModel.generate_model_from_config(self.model_config).to(self.device)

        self.init_log()
        self.save_updated_config()

    def update_config(self):
        time = datetime.now()
        timestamp = datetime.strftime(time, '%Y%m%d-%H:%M:%S')
        model_folder_name = make_config_string(self.config['model']) + '_' + timestamp

        self.log_folder = create_folder(self.config['base_dir'], model_folder_name)
        self.model_folder = create_folder(self.log_folder, 'models')

        self.model_config['saved_model_dir'] = os.path.join(self.model_folder, 'saved_model')
        self.model_config['saved_log_dir'] = os.path.join(self.log_folder, 'log')

        model_name = self.model_config['name']
        self.config['output_config_dir'] = os.path.join('configs', f'{model_name}_output.yaml')

        return

    def init_log(self):
        self.log = LogWriter(self.model_config['saved_log_dir'], self.config)
        # use self.args not args, cuz init function may add things to args
        self.log.initBest()

        return

    def save_updated_config(self):
        save_config(self.config['output_config_dir'], self.config)
        return

    def get_dataloader(self, data_filename='data_dir', dataset_fn=TPPDataset, **kwargs):
        loaders = []
        splits = ["train", 'dev', 'test']
        event_types = None
        token_types = 0
        for _split in splits:
            with open(os.path.join(self.data_config[data_filename].format(_split)), "rb") as f_in:
                # latin-1 for GaTech data
                try:
                    _data = pickle.load(f_in, encoding='latin-1')
                except:
                    _data = pickle.load(f_in)
                if event_types is None:
                    event_types = _data["dim_process"]
                else:
                    assert _data["dim_process"] == event_types, "inconsistent dim_process in different splits?"
                dataset = dataset_fn(_data[_split], event_types, concurrent=False, add_bos=False, add_eos=False,
                                     skip_padding=kwargs.get('skip_padding', False))
                assert dataset.event_num <= event_types, f"{_split}.pkl has more event types than specified in dim_process!"
                token_types = max(token_types, dataset.num_types)
                loaders.append(create_dataloader(dataset, batch_size=self.data_config['batch_size']))
        assert token_types > event_types, f"at least we should include [PAD]! token: {token_types}, event: {event_types}"
        return loaders

    def run_one_epoch(self, model, dataLoader, mode, optimizer=None):
        assert mode in {"train", "eval"}
        if mode == "eval":
            model = model.eval()
        else:
            assert optimizer is not None
        total_log_like = 0
        total_acc = 0
        total_event_ll, total_non_event_ll = 0, 0
        num_tokens = 0
        pad_idx = self.train_loader.dataset.pad_index
        num_events = 0
        all_logs = []
        all_logs_token = []
        all_type_ll_token = []
        all_time_ll_token = []
        for batch in tqdm(dataLoader, mininterval=2, desc=f'   - ({mode}) -    ', leave=False):
            new_batch = [x.to(self.device) for x in batch]
            time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = new_batch
            event_ll, non_event_ll, enc_inten = model.compute_loglik(new_batch)
            if hasattr(self.train_config, "ignore_first"):
                if self.train_config['ignore_first']:
                    non_event_ll[:, 0] *= 0
            _batch_loss = event_ll.sum(dim=-1) - non_event_ll.sum(dim=-1)
            _loss = -torch.sum(_batch_loss)
            if mode == "train":
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_log_like += -_loss.item()
            total_event_ll += event_ll.sum().item()
            total_non_event_ll += non_event_ll.sum().item()
            type_lls = event_ll - torch.log(enc_inten.sum(dim=-1) + model.eps)
            time_lls = event_ll - non_event_ll - type_lls
            total_acc += ((torch.argmax(enc_inten, dim=-1) == event_seq[:, 1:]) * batch_non_pad_mask[:, 1:]).sum()
            num_tokens += event_seq[:, 1:].ne(pad_idx).sum().item()
            num_events += (event_seq[:, 1:] < pad_idx).sum().item()
            all_logs_token.extend([(x, 1.0) for x in (event_ll - non_event_ll)[batch_non_pad_mask[:, 1:]].tolist()])
            all_type_ll_token.extend([(x, 1.0) for x in type_lls[batch_non_pad_mask[:, 1:]].tolist()])
            all_time_ll_token.extend([(x, 1.0) for x in time_lls[batch_non_pad_mask[:, 1:]].tolist()])
            all_logs.extend([(x, y) for x, y in zip(_batch_loss.tolist(), event_seq.ne(pad_idx).sum(dim=-1).tolist())])
        return total_log_like, total_acc / num_tokens, (total_event_ll, total_non_event_ll), \
               num_tokens, num_events, all_logs, all_logs_token, \
               all_type_ll_token, all_time_ll_token

    def create_thinningsampler(self, num_sample, num_exp):
        self.thinning_sampler = EventSampler(num_sample, num_exp, device=self.device)

    def run_prediction(self, model, data_loader):
        self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        verbose = self.config['verbose']
        thinning_sampler = self.thinning_sampler
        for _batch in tqdm(data_loader, desc=f"   (Pred)    ", leave=False, mininterval=2):
            time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = _batch
            # thinning can only run in single instance mode, not in batch mode
            num_batch = time_seq.size(0)
            for i in range(num_batch):
                rst = []
                _time_seq, _event_seq = time_seq[i][batch_non_pad_mask[i]], event_seq[i][batch_non_pad_mask[i]]
                seq_len = _time_seq.size(0)
                duration = _time_seq[-1].item() + np.finfo(float).eps
                num_sub = seq_len - 1
                for j in range(seq_len - 1):
                    next_event_name, next_event_time = _event_seq[j + 1].item(), _time_seq[j + 1].item()
                    current_event_name, current_event_time = _event_seq[j].item(), _time_seq[j].item()
                    time_last_event = _time_seq[j].item()
                    if verbose:
                        print(
                            f"for {seq_id}-th seq, predict after {j}-th event {current_event_name} at {current_event_time:.4f}")
                    next_event_dtime = next_event_time - time_last_event
                    avg_future_dtime = (duration - time_last_event) / (num_sub - j)
                    look_ahead = max(next_event_dtime, avg_future_dtime)
                    boundary = time_last_event + 4 * look_ahead
                    _event_prefix, _time_prefix = _event_seq[:j + 1].unsqueeze(0).cuda(), _time_seq[:j + 1].unsqueeze(
                        0).cuda()
                    accepted_times, weights = thinning_sampler.draw_next_time(
                        [[_event_prefix, _time_prefix],
                         time_last_event, boundary, model]
                    )
                    time_uncond = float(torch.sum(accepted_times * weights))
                    dtime_uncond = time_uncond - time_last_event
                    intensities_at_times = model.compute_intensities_at_sampled_times(
                        _event_prefix, _time_prefix,
                        _time_seq[j + 1].reshape(1, 1)
                    )[0, 0]
                    top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)
                    # since we use int to represent event names already
                    top_event_names = [int(top_i) for top_i in top_ids]
                    rst.append(
                        (
                            time_uncond, dtime_uncond, top_event_names,
                            next_event_time, next_event_dtime, next_event_name
                        )
                    )
                    if verbose:
                        print(
                            f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                        print(
                            f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
                results.append(rst)
                seq_id += 1
        return results

    def run_generation_multi_step(self, model, start_time, start_event, start_dtime,
                                  sample_len, event_num, patience, num_sample_seqs=1, max_time=10,
                                  save_in_pkl=True):
        """
        The code is the same as the one in ar_model.gen.py.
        There must be a better way to encapsulate it.
        """
        thinning_sampler = self.thinning_sampler
        result = []
        # thinning can only run in single instance mode, not in batch mode
        for _ in range(num_sample_seqs):
            types = start_event.copy()
            times = start_time.copy()
            dtimes = start_dtime.copy()
            intens = [[0] * event_num]
            for _ in range(sample_len):
                _time_seq, _event_seq = torch.tensor(times, dtype=torch.float64), torch.LongTensor(types)
                time_last_event = _time_seq[-1].item()
                boundary = time_last_event + max_time
                _event_prefix, _time_prefix = _event_seq.unsqueeze(0).to(self.device), _time_seq.unsqueeze(0).to(
                    self.device)
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     time_last_event, boundary, model, patience],
                    mode="unbiased"
                )
                _time_uncond = torch.sum(accepted_times * weights)
                time_uncond = float(_time_uncond)
                dtime_uncond = time_uncond - time_last_event
                times.append(time_uncond)
                dtimes.append(dtime_uncond)
                intensities_at_times = model.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    _time_uncond.reshape(1, 1)
                )[0, 0]
                intens.append(intensities_at_times.tolist())
                next_event_name = torch.multinomial(intensities_at_times, 1)[0].item()
                types.append(next_event_name)
            if save_in_pkl:
                rst = [
                    {
                        "time_since_start": times[x],
                        "type_event": types[x],
                        "time_since_last_event": dtimes[x],
                        "intensities": intens[x]
                    } for x in range(len(times))
                ]
            else:
                rst = [times, types, dtimes]
            result.append(rst)
        return result
