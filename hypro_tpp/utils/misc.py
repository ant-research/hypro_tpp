import os


def make_config_string(config, max_num_key=4):
    """ Generate a name for config files """
    str_config = ''
    num_key = 0
    for k, v in config.items():
        if num_key < max_num_key:  # for the moment we only record model name
            if k == 'name':
                str_config += str(v) + '_'
                num_key += 1
    return str_config[:-1]


def create_folder(*args):
    """Create path if the folder doesn't exist.
    :param args:
    :return: The folder's path depends on operating system.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def setup_seed(seed=9899):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
