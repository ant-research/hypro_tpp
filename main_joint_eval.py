import argparse
import numpy as np
from functools import reduce
from hypro_tpp import JointEval
from hypro_tpp.utils import load_config
from hypro_tpp.utils.misc import setup_seed


def main(trial_id=1):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=False, default='configs/taxi/joint_eval.yaml',
                        help='Configuration filename for hypro_tpp or restoring the model.')

    args = parser.parse_args()

    config = load_config(args.config)

    model_runner = JointEval(config)

    metrics = model_runner.run(trial_id)
    return metrics


if __name__ == '__main__':
    setup_seed()
    max_trials = 5
    metric_list = []
    for i in range(max_trials):
        print(f'------------------ CTTX: {i + 1} ------------------')
        metric = main(i + 1)
        metric_list.append(metric)

    print(f'----------- Average of CTTX {max_trials} runs ------------')
    for key in metric_list[0].keys():
        val = reduce(lambda x, y: x + y, [metric[key] for metric in metric_list]) / max_trials
        if type(val) in [np.ndarray, list, set]:
            val = ["%.4f" % i for i in val]
        print(f'{key}:\t{val}')