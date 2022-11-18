import torch
import numpy as np
import pandas as pd
from hypro_tpp.model_runner import ModelRunner
from hypro_tpp.utils import distance_between_event_seq, count_mae
from hypro_tpp.models import BaseModel


class JointEval(ModelRunner):

    def __init__(self, config):
        super(JointEval, self).__init__(config)
        self.eval_config = config.get('eval', None)
        self.eval_sample_limit = self.eval_config.get('eval_sample_limit', 2 ** 32 - 1)
        self.generator_config = self.model_config['generator']
        self.discriminator_config = self.model_config['discriminator']
        self.eval_del_cost = self.eval_config['del_cost']

        self.num_samples = self.generator_config['thinning_params']['num_samples']
        self.num_seqs = self.generator_config['thinning_params']['num_seqs']
        self.num_exp = self.generator_config['thinning_params']['num_exp']
        self.patience = self.generator_config['thinning_params']['patience']
        self.event_num = self.model_config['num_event_types_no_pad']
        self.create_thinningsampler(self.num_samples, self.num_exp)

        self.update_model_config(self.generator_config)
        self.generator = BaseModel.generate_model_from_config(self.generator_config).to(self.device)
        self.generator.load_state_dict(torch.load(self.generator_config['pretrained_model_dir']), strict=False)
        self.update_model_config(self.discriminator_config)
        self.discriminator = BaseModel.generate_model_from_config(self.discriminator_config).to(self.device)
        self.discriminator.load_state_dict(torch.load(self.discriminator_config['pretrained_model_dir']))

        count = sum(p.numel() for p in self.generator.parameters())
        print('Generator parameters', count)

        count = sum(p.numel() for p in self.discriminator.parameters())
        print('discriminator parameters', count)

        [self.train_loader, self.dev_loader, self.test_loader] \
            = self.get_dataloader()

    def update_model_config(self, config_dict):
        key_to_add = ['add_bos', 'num_event_types_pad', 'num_event_types_no_pad', 'event_pad_index']
        config_dict.update({k: self.model_config[k] for k in key_to_add})
        return

    def run(self, trial_id):
        if self.eval_config['target_set'] == "train":
            data_loader = self.train_loader
        elif self.eval_config['target_set'] == "dev":
            data_loader = self.dev_loader
        else:
            data_loader = self.test_loader

        metrics = self.run_one_epoch(data_loader, trial_id)

        return metrics

    @staticmethod
    def retrieve_prediction(sample):
        time_pred = [sample_[0] for sample_ in sample]
        event_pred = [sample_[1] for sample_ in sample]
        dtime_pred = [sample_[2] for sample_ in sample]
        return torch.tensor(time_pred, dtype=torch.float64), torch.tensor(dtime_pred,
                                                                          dtype=torch.float64), torch.LongTensor(
            event_pred)

    def eval_distance(self, ground_truth_tuple, sample_tuple):
        distance = [distance_between_event_seq(ground_truth_tuple,
                                               sample_tuple,
                                               del_cost=[del_cost],
                                               trans_cost=1.0,
                                               num_types=self.event_num)[0][0] for del_cost in self.eval_del_cost]

        return distance

    def eval_mae(self, ground_truth_tuple, sample_tuple):
        def most_frequent(list_):
            return max(set(list_), key=list_.count)

        most_freq_type = most_frequent(ground_truth_tuple[1])

        mae = count_mae(ground_truth_tuple,
                        sample_tuple,
                        most_freq_type)

        return mae

    def eval_acc(self, ground_truth_tuple, sample_tuple):
        event_label = ground_truth_tuple[1]
        event_pred = sample_tuple[1].cpu().numpy()
        min_len = min(len(event_pred), len(event_label))
        if len(event_pred) == 0 and len(event_label) != 0:
            return 0
        elif len(event_pred) != 0 and len(event_label) == 0:
            return 1
        return np.equal(event_pred[:min_len], event_label[:min_len]).mean()

    def eval_rmse(self, ground_truth_tuple, sample_tuple):
        time_label = ground_truth_tuple[0]
        time_pred = sample_tuple[0].cpu().numpy()
        return np.sqrt(np.mean(time_label - time_pred) ** 2)

    def filter_points(self, ground_truth_tuple, sample_tuple, all_tuple):
        time_range = self.config['time_range']

        filter_max_time = ground_truth_tuple[0][0] + time_range
        horizon = len(ground_truth_tuple[0])

        def _truncate_tuple(one_tuple):
            end_i = horizon
            for i in range(horizon):
                if one_tuple[0][i] > filter_max_time:
                    end_i = i
                    break

            return one_tuple[0][:end_i], one_tuple[1][:end_i]

        # filter ground truth
        ground_truth_tuple = _truncate_tuple(ground_truth_tuple)
        sample_tuple = _truncate_tuple(sample_tuple)
        all_tuple = [_truncate_tuple(one_tuple) for one_tuple in all_tuple]

        return ground_truth_tuple, sample_tuple, all_tuple

    def eval_l2_error(self, ground_truth_tuple, sample_tuple):
        label_type = ground_truth_tuple[1]
        pred_type = sample_tuple[1]

        def _type_count_vector(seq_type):
            vector = np.zeros(self.config['model']['num_event_types_no_pad'])
            for k in seq_type:
                vector[k] = vector[k] + 1
            return vector

        label_vector = _type_count_vector(label_type)
        pred_vector = _type_count_vector(pred_type)

        # in the code i have not divided it by the num of event types.
        l2_error = np.sum((label_vector - pred_vector) ** 2) ** 0.5 
        return l2_error

    def evaluation_per_seq(self, ground_truth_tuple, sample_tuple, all_tuple):
        res = dict()

        ground_truth_tuple, sample_tuple, all_tuple = self.filter_points(ground_truth_tuple, sample_tuple, all_tuple)

        # distance evaluation
        distance = self.eval_distance(ground_truth_tuple, sample_tuple)
        res['joint_distance'] = distance

        all_distance = [self.eval_distance(ground_truth_tuple, pred_tuple) for pred_tuple in all_tuple]
        res['all_distance'] = all_distance

        mae = self.eval_mae(ground_truth_tuple, sample_tuple)
        res['joint_mae'] = mae

        all_mae = [self.eval_mae(ground_truth_tuple, pred_tuple) for pred_tuple in all_tuple]
        res['all_mae'] = all_mae

        acc = self.eval_acc(ground_truth_tuple, sample_tuple)
        res['joint_acc'] = acc

        res['joint_l2_error'] = self.eval_l2_error(ground_truth_tuple, sample_tuple)

        all_acc = [self.eval_acc(ground_truth_tuple, pred_tuple) for pred_tuple in all_tuple]
        res['all_acc'] = all_acc

        all_l2_error = [self.eval_l2_error(ground_truth_tuple, pred_tuple) for pred_tuple in all_tuple]
        res['all_l2_error'] = all_l2_error

        return res

    def eval_oracle_distance(self, metrics_per_seq):
        seq_id = metrics_per_seq.keys()
        all_distance = [metrics_per_seq[id]['all_distance'] for id in seq_id]
        oracle_distance_per_seq = [np.array(distance).min(axis=0).tolist() for distance in all_distance]
        oracle_distance_avg_seq = np.array(oracle_distance_per_seq).mean(axis=0)
        return oracle_distance_avg_seq

    def eval_joint_distance(self, metrics_per_seq):
        seq_id = metrics_per_seq.keys()
        joint_distance = [metrics_per_seq[id]['joint_distance'] for id in seq_id]
        joint_distance_avg_seq = np.array(joint_distance).mean(axis=0)

        return joint_distance_avg_seq

    def evaluation_average_by_seq(self, metrics_per_seq, save_res=False):
        seq_id = list(metrics_per_seq.keys())

        res_metrics = dict()
        desired_scalar_metrics = ['joint_acc', 'all_acc', 'joint_l2_error', 'all_l2_error']
        for metric_name in desired_scalar_metrics:
            each_metric = [np.mean(metrics_per_seq[tmp_id][metric_name]) for tmp_id in seq_id]
            res_metrics[metric_name] = np.mean(each_metric)

        desired_vector_metrics = ['joint_distance', 'all_distance']
        for metric_name in desired_vector_metrics:
            dim = np.array(metrics_per_seq[seq_id[0]][metric_name]).shape[-1]
            tmp = np.array([val for tmp_id in seq_id for val in metrics_per_seq[tmp_id][metric_name]])
            tmp = tmp.reshape([-1, dim])
            res_metrics[metric_name] = tmp.mean(axis=0)

        return res_metrics

    def make_discriminator_sample(self, type_pred, time_pred, batch_non_pad_mask, attention_mask):
        num_noise_sample = type_pred.size()[0]
        batch_non_pad_mask_ = batch_non_pad_mask.repeat(num_noise_sample, 1)
        attention_mask_ = attention_mask.repeat(num_noise_sample, 1, 1)

        return (type_pred.to(self.device), time_pred.to(self.device), batch_non_pad_mask_.to(self.device),
                attention_mask_.to(self.device))

    def multi_step_pred(self, time_seq_history, event_seq_history, dtime_seq_history, sample_len, num_sample_seqs):
        sample = self.run_generation_multi_step(self.generator,
                                                time_seq_history,
                                                event_seq_history,
                                                dtime_seq_history,
                                                event_num=self.event_num,
                                                sample_len=sample_len,
                                                patience=self.patience,
                                                num_sample_seqs=num_sample_seqs,
                                                save_in_pkl=False)
        # [num_samples, seq_len]
        time_pred, dtime_pred, type_pred = self.retrieve_prediction(sample)

        return time_pred, dtime_pred, type_pred, sample

    def run_one_epoch(self, data_loader, trial_id):
        import os
        import pickle

        fn_template = self.config['gen_sample_file_format']
        sample_fn = fn_template.format(self.data_config['data_name'], trial_id)

        ns = self.eval_config['num_samples_per_iteration']
        sample_len = self.model_config['sample_len']

        #
        attention_masks = []
        batch_non_pad_masks = []
        max_n = max(ns)
        scores = []
        time_labels = []
        event_seq_labels = []
        time_preds = []
        type_preds = []
        seq_id = 0
        batch_id = 0

        if not os.path.exists(sample_fn):
            for batch in data_loader:
                batch_id += 1
                if batch_id > self.eval_sample_limit:
                    break
                time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
                # thinning can only run in single instance mode, not in batch mode
                batch_size = time_seq.size(0)
                for i in range(batch_size):
                    _time_seq, _dtime_seq, _event_seq = time_seq[i][batch_non_pad_mask[i]], \
                                                        time_delta_seq[i][batch_non_pad_mask[i]], \
                                                        event_seq[i][batch_non_pad_mask[i]]
                    # if sample_len >= seq_len, then we skip it
                    if sample_len >= len(_time_seq):
                        continue

                    seq_id += 1

                
                    # they are already known input, not for prediction
                    time_seq_history = _time_seq[:-sample_len].numpy().tolist()
                    event_seq_history = _event_seq[:-sample_len].numpy().tolist()
                    dtime_seq_history = _dtime_seq[:-sample_len].numpy().tolist()

                    time_seq_label = _time_seq[-sample_len:].numpy().tolist()
                    event_seq_label = _event_seq[-sample_len:].numpy().tolist()

                    # discriminator sampling
                    # run sampling, each sample is a predicted multi-step sequence
                    # [num_samples, seq_len]
                    time_pred, dtime_pred, type_pred, sample = self.multi_step_pred(time_seq_history,
                                                                                    event_seq_history,
                                                                                    dtime_seq_history,
                                                                                    sample_len,
                                                                                    max_n)
                    time_labels.append(time_seq_label)
                    event_seq_labels.append(event_seq_label)
                    time_preds.append(time_pred)
                    type_preds.append(type_pred)
                    batch_non_pad_masks.append(batch_non_pad_mask)
                    attention_masks.append(attention_mask)

            # save samples into file
            res_dict = dict()
            res_dict['time_labels'] = time_labels
            res_dict['event_seq_labels'] = event_seq_labels
            res_dict['time_preds'] = time_preds
            res_dict['type_preds'] = type_preds
            res_dict['batch_non_pad_masks'] = batch_non_pad_masks
            res_dict['attention_masks'] = attention_masks

            # if parent dir not existed
            parent_dir = os.path.dirname(sample_fn)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)


            pickle.dump(res_dict, open(sample_fn, 'wb'))
        else:
            print('Load from file:', sample_fn)
            res_dict = pickle.load(open(sample_fn, 'rb'))
            time_labels = res_dict['time_labels']
            event_seq_labels = res_dict['event_seq_labels']
            time_preds = res_dict['time_preds']
            type_preds = res_dict['type_preds']
            batch_non_pad_masks = res_dict['batch_non_pad_masks']
            attention_masks = res_dict['attention_masks']

        discriminator = self.discriminator.eval()
        for type_pred, time_pred, batch_non_pad_mask, attention_mask in zip(type_preds, time_preds, batch_non_pad_masks,
                                                                            attention_masks):
            # compute energy scores
            discriminator_sample = self.make_discriminator_sample(type_pred,
                                                                  time_pred,
                                                                  batch_non_pad_mask,
                                                                  attention_mask)
            # [num_samples, seq_len, hidden_size]
            seq_logits = discriminator.forward(*discriminator_sample)
            # [num_samples, seq_len, hidden_size]
            batch_non_pad_mask_ = batch_non_pad_mask.repeat(time_pred.size()[0], 1).to(self.device)
            logits = discriminator.get_logits_at_last_step(seq_logits, batch_non_pad_mask_)
            logits = discriminator.predict_as_discriminator(logits)
            
            score = -logits[:, 0]
            scores.append(score)

        metrics = None
        for n in ns:
            print(f'n: {n}')
            res = dict()
            seq_id = 0

            ground_truth_tuple_list = []
            pred_sample_tuple_list = []
            all_tuple_list = []

            length = len(scores)
            n_folders = 1
            for i in range(n_folders):
                start_i = int(i / n_folders * length)
                end_i = int((i + 1) / n_folders * length)
                print('------ Fold', i)
                for j, (score, time_seq_label, event_seq_label, time_pred, type_pred) in enumerate(zip(scores, time_labels,
                                                                                        event_seq_labels, time_preds,
                                                                                        type_preds)):
                    if j < start_i or j >= end_i:
                        continue
                    # different number samples
                    score = score[:n]
                    # time_seq_label = time_seq_label[:n]
                    # event_seq_label = event_seq_label[:n]
                    time_pred = time_pred[:n]
                    type_pred = type_pred[:n]

                    # compute sampling proba
                    probs = torch.softmax(score, -1).cpu().detach().numpy()
                    probs = probs / probs.sum()

                    # resample
                    sample_set = np.arange(n)
                    
                    sample_idx = probs.argmax()
                    
                    # re-eval
                    ground_truth_tuple = (time_seq_label, event_seq_label)

                    # joint model tuple
                    joint_tuple = (time_pred[sample_idx][-sample_len:], type_pred[sample_idx][-sample_len:])

                    # all tuples from joint sampling
                    all_tuple = [(time_pred[sample_idx][-sample_len:], type_pred[sample_idx][-sample_len:])
                                 for sample_idx in sample_set]

                    res[seq_id] = self.evaluation_per_seq(ground_truth_tuple,
                                                          joint_tuple,
                                                          all_tuple)

                    ground_truth_tuple_list.append(
                        ground_truth_tuple
                    )
                    pred_sample_tuple_list.append(joint_tuple)
                    all_tuple_list.append(all_tuple)

                    seq_id += 1

                # min distance for each pred seq
                metrics = self.evaluation_average_by_seq(res, save_res=True)

                if n == 1:
                    print(' baseline ')
                print('All metrics:')
                for key, val in metrics.items():
                    if type(val) in [np.ndarray, list, set]:
                        val = ["%.4f" % i for i in val]
                    print(f'\t{key}:\t{val}')

        return metrics