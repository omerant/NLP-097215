import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import re


def get_training_info(weights_path='weights'):
    weight_file_names = sorted(os.listdir(weights_path))
    training_info = OrderedDict()  # dict of form {reg_lambda: [[num_iters], [likelihood]]}
    for w_file_name in weight_file_names:
        path_to_file = os.path.join(weights_path, w_file_name)
        if os.path.isdir(path_to_file):
            continue
        info = w_file_name.split('-')
        feature_version = int(info[0])
        reg_lambda = float(info[1].split('=')[1])

        num_iter = int(info[2].split('=')[1])
        batch_size = int(info[2].split('=')[1])
        with open(path_to_file, 'rb') as f:
            optimal_params = pickle.load(f)
        likelihood = -optimal_params[1]

        if not training_info.get(reg_lambda, None):
            training_info[reg_lambda] = [[], []]

        training_info[reg_lambda][0].append(num_iter)
        training_info[reg_lambda][1].append(likelihood)

        v = optimal_params[0]
        print(f'feature version={feature_version}, {reg_lambda}, {batch_size}, {num_iter}, '
              f'likelihood={likelihood}')
    return training_info


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


def is_number(s):
    return bool(re.match(r'^-?\d+(?:\,\d+)?$', s)) or bool(re.match(r'^-?\d+(?:\.\d+)?$', s))
