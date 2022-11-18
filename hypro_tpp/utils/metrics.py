import numpy as np
import pickle


def type_acc_np(preds, labels, **kwargs):
    """ Type accuracy ratio  """
    type_pred = preds
    type_label = labels

    return np.mean(type_pred == type_label)


def time_rmse_np(preds, labels, **kwargs):
    """ RMSE for time predictions """
    seq_mask = kwargs.get('seq_mask', np.isreal(np.ones_like(preds)))
    dt_pred = preds['dtimes'][seq_mask]
    dt_label = labels['dtimes'][seq_mask]

    rmse = np.sqrt(np.mean((dt_pred - dt_label) ** 2))
    return rmse


# ref: https://github.com/hongyuanmei/neural-hawkes-particle-smoothing/blob/8f33c75038e739a2a0b61db854dd97d918ce2d19/nhps/distance/utils/edit_distance.py
def find_alignment_mc(seq1, seq2, del_cost, trans_cost):
    """
    We use dynamic programming to find the best alignments between two seqs.
    ``nc'' means that this functions support a series of del_cost values.
    Note: Not support multiple types.
    :param np.ndarray seq1: Time stamps of seq #1.
    :param np.ndarray seq2: Time stamps of seq #2.
    :param np.ndarray del_cost: A series of delete cost.
    :param float trans_cost: Transportation cost per unit length.
    :return: Alignment list and minimum distances for all the del_cost values.
    """
    n_cost = len(del_cost)
    n1 = len(seq1)
    n2 = len(seq2)
    # shape=[n2, n1]
    trans_mask = np.abs(seq2.repeat(n1).reshape(n2, n1) - seq1) * trans_cost
    # shape=[n1+1, n1+1]
    del_mask = np.arange(n1 + 2, dtype=np.float32) \
                   .repeat(n1 + 1).reshape(n1 + 2, n1 + 1) \
                   .T.reshape(-1)[:(n1 + 1) ** 2].reshape(n1 + 1, n1 + 1) - 1
    del_mask[np.tril_indices(n1 + 1, -1)] = float('inf')
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.repeat(n_cost).reshape(n1 + 1, n1 + 1, n_cost) * del_cost
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.transpose([1, 0, 2]).copy()
    # shape=[n1+1, n_cost]
    overhead = np.empty(shape=[n1 + 1, n_cost], dtype=np.float32)
    overhead.fill(float('inf'))
    overhead[0, :] = 0.0
    # shape=[n2, n1+1, n_cost]
    back_pointers = np.empty(shape=[n2, n1 + 1, n_cost], dtype=np.int32)
    for n2_idx in range(n2):
        # shape=[n1+1, n1+1, n_cost]
        add_mask = del_mask.copy()
        add_mask[1:, :, :] += np.outer(trans_mask[n2_idx],
                                       np.ones(shape=[(n1 + 1) * n_cost],
                                               dtype=np.float32)).reshape(n1, n1 + 1, n_cost)
        add_mask[np.arange(n1 + 1), np.arange(n1 + 1), :] = del_cost
        # shape=[n1+1, n1+1, n_cost]
        cost_mat = overhead + add_mask
        # shape=[n1+1, n_cost]
        choices = np.argmin(cost_mat, axis=1)
        back_pointers[n2_idx] = choices
        overhead = cost_mat.min(axis=1)
    overhead += np.outer(np.arange(n1, -1, -1, dtype=np.float32), np.ones(shape=[n_cost])) * del_cost
    # shape=[n_cost]
    curr_choice = np.argmin(overhead, axis=0)
    # shape=[n_cost]
    min_distance = overhead.min(axis=0)
    best_route = [curr_choice]
    # shape=[n1+1, n_cost]
    for choice_list in back_pointers[::-1]:
        # shape=[n_cost]
        curr_choice = choice_list[curr_choice, np.arange(n_cost)]
        best_route.append(curr_choice)
    # shape=[n2, n_cost]
    best_route = np.array(best_route)

    align_pairs = list()
    for cost_idx in range(n_cost):
        best_route_ = best_route[:, cost_idx]
        pairs = list()
        memo = -1
        for n2_idx_plus_1, choice_made in enumerate(best_route_[::-1]):
            if choice_made != memo:
                pairs.append([choice_made - 1, n2_idx_plus_1 - 1])
            memo = choice_made
        align_pairs.append(pairs[1:])

    return [align_pairs,  # len=n_cost
            min_distance  # shape=[n_cost]
            ]


def find_alignment(seq1, seq2, del_cost, trans_factor):
    """
    Similar functionality with find_alignment_nc, but for single del_cost cost.
    :param np.ndarray seq1:
    :param np.ndarray seq2:
    :param float del_cost:
    :param float trans_factor:
    :return:
    """
    align_pairs, min_distance = \
        find_alignment_mc(seq1, seq2, np.array([del_cost]), trans_factor)
    return align_pairs[0], float(min_distance[0])


def float_equal(a, b):
    eps = 1e-4
    return (1 - eps) < (a / b) < (1 + eps)


def count_mae(ref_seq, decode_seq, target_type, obs_period_start=None, obs_period_end=None):
    """
    Args:
        ref_seq: [time_seqs, event_seqs]
        decode_seq: [time_seqs, event_seqs]
        target_type: index of event type
        obs_period_start: start timestamps of the test period
        obs_period_end: end timestamps of the test period

    Returns:

    """

    def extract_count(time_seq, event_seq, obs_period_start, obs_period_end):
        event_seq_ = event_seq[(time_seq >= obs_period_start) & (time_seq <= obs_period_end)]
        event_seq_ = event_seq_[event_seq_ == target_type]
        return len(event_seq_)

    ref_time_seqs, ref_event_seqs = ref_seq
    decode_time_seqs, decode_event_seqs = decode_seq
    if obs_period_start is None:
        obs_period_start = ref_time_seqs[0]
    if obs_period_end is None:
        obs_period_end = ref_time_seqs[-1]

    label = extract_count(np.array(ref_time_seqs), np.array(ref_event_seqs), obs_period_start, obs_period_end)
    pred = extract_count(np.array(decode_time_seqs), np.array(decode_event_seqs), obs_period_start, obs_period_end)

    return abs(pred - label) / (label + 1e-5)


def distance_between_event_seq(ref_seq, decode_seq, del_cost, trans_cost, num_types):
    """
    Args:
        ref_seq: [time_seqs, event_seqs]
        decode_seq: [time_seqs, event_seqs]
        del_cost:
        trans_cost:
        num_types:

    Returns:

    """
    num_cost = len(del_cost)

    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    seq_per_types = [[list(), list()] for _ in range(num_types)]
    for seq_idx, seq in enumerate([ref_seq, decode_seq]):
        for event_time, event_type in zip(*seq):
            if event_type >= num_types:
                continue
            seq_per_types[event_type][seq_idx].append(event_time)

    for type_idx in range(num_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(
            ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost \
                                          - del_cost[cost_idx] * (n_ins_per_cost + n_del_per_cost)

    return distances, total_trans_cost, num_true, num_del, num_ins, num_align


def edit_distance_mt_mc(ref, decoded, del_cost, trans_cost, n_types):
    """
    :param list ref:
    :param list decoded:
    :param np.ndarray del_cost: 越高越倾向于移动
    :param float trans_cost:
    :param int n_types:
    """
    num_cost = len(del_cost)

    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    ref = ref[:5]
    decoded = decoded[:3]
    # print(ref)
    # print(decoded)

    seq_per_types = [[list(), list()] for _ in range(n_types)]
    for seq_idx, seq in enumerate([ref, decoded]):
        print(seq_idx)
        print(seq)
        for token in seq:
            event_type = token['type_event']
            if event_type >= n_types:
                continue
            seq_per_types[event_type][seq_idx].append(token['time_since_start'])

    for type_idx in range(n_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(
            ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost \
                                          - del_cost[cost_idx] * (n_ins_per_cost + n_del_per_cost)

    return distances, total_trans_cost, num_true, num_del, num_ins, num_align
