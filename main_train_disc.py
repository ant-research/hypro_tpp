import argparse
from hypro_tpp import DiscriminatorTrainer
from hypro_tpp.utils import load_config
from hypro_tpp.utils.misc import setup_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=False, default='configs/taxi/attnhp_disc_bce.yaml',
                        help='Configuration filename for hypro_tpp or restoring the model.')

    args = parser.parse_args()

    config = load_config(args.config)

    model_runner = DiscriminatorTrainer(config)

    model_runner.run()


if __name__ == '__main__':
    setup_seed()
    main()
