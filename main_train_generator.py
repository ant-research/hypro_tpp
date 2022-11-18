import argparse
from hypro_tpp import GeneratorTrainer
from hypro_tpp.utils import load_config
from hypro_tpp.utils.misc import setup_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=False, default='configs/taxi/attnhp_train.yaml',
                        help='Configuration filename for hypro_tpp or restoring the model.')

    args = parser.parse_args()

    config = load_config(args.config)

    model_runner = GeneratorTrainer(config)

    model_runner.run()


if __name__ == '__main__':
    setup_seed()
    main()
