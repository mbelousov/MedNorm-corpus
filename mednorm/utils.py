import os
import shutil

import numpy as np
import pandas as pd
import yaml
from colorclass import Color


def read_multilabel_dataset(input_file, input_col=2, target_col=(-2, -1),
                            encoding='utf-8'):
    df = pd.read_csv(input_file, sep='\t',
                     encoding=encoding, dtype='str')
    if not (isinstance(target_col, list) or isinstance(target_col, tuple)):
        target_col = [target_col]
    target_col = [c if isinstance(c, str) else df.columns[c]
                  for c in target_col]

    df = df.dropna(subset=target_col)

    texts = np.asarray(df[df.columns[input_col]].values)
    labels = np.asarray([np.asarray([lbls.split(" ") for lbls in df[c].values])
                         for c in target_col])
    return texts, labels


def print_success(message):
    print(Color("{green}%s{/green}" % message))


def print_bold(message):
    print('\x1b[1;37m' + message.strip() + '\x1b[0m')


def normpath(filepath):
    filepath = os.path.expandvars(os.path.expanduser(filepath))
    filepath = os.path.normpath(filepath)
    if not os.path.isabs(filepath):
        filepath = os.path.abspath(filepath)
    return filepath


def makedirs(dirpath, remove=False):
    if remove and os.path.exists(dirpath):
        shutil.rmtree(dirpath)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def makedirs_file(filepath, remove=False):
    folder_path = os.path.dirname(normpath(filepath))
    makedirs(folder_path, remove=remove)


def load_yaml(filepath):
    with open(normpath(filepath), 'r') as fp:
        return yaml.load(fp)


def normlist(values, sep=","):
    if isinstance(values, str):
        return [v.strip() for v in values.split(sep)]
    elif isinstance(values, int) or isinstance(values, float):
        return [values]
    return list(values)


def flatten(l):
    return [i for sl in l for i in sl]


def flatten_list(y):
    if isinstance(y, list) or isinstance(y, np.ndarray):
        return flatten(y)
    return y


def remove_list_indices(l, indices):
    return [item for i, item in enumerate(l) if i not in indices]


def remove_list_values(l, vals):
    return [item for item in l if item not in vals]


def filter_list(l, indices):
    return [l[i] for i in indices]
