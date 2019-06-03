from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

from mednorm.utils import filter_list, flatten_list, remove_list_values


class LabelEncoder(object):
    multi_label = False

    def __init__(self):
        super(LabelEncoder, self).__init__()

    def fit(self, X, y=None):
        raise NotImplementedError("not implemented")

    def item_transform(self, item):
        raise NotImplementedError("not implemented")

    @property
    def classes_(self):
        raise NotImplementedError("not implemented")

    @property
    def size(self):
        raise NotImplementedError("not implemented")

    def inverse_transform(self, X):
        raise NotImplementedError("not implemented")

    def decode_index(self, idx, default=None):
        raise NotImplementedError("not implemented")

    def label_index(self, label):
        raise NotImplementedError("not implemented")

    @property
    def multi_label_flag(self):
        return self.multi_label


class MultiLabelReducer(LabelEncoder):
    def __init__(self):
        self._sp_cache = {}
        self._frq = None
        self._g = nx.Graph()
        super(MultiLabelReducer, self).__init__()

    def fit(self, X, y=None):
        for labels in X:
            for lbl in labels:
                self._g.add_node(lbl)
                for lbl2 in labels:
                    if lbl2 == lbl:
                        continue
                    self._g.add_edge(lbl, lbl2)

        all_labels = [lbl for labels in X for lbl in labels]
        self._frq = dict(Counter(all_labels).items())
        self._sp_cache = {}
        return self

    def transform(self, X):
        Xt = []
        for labels in X:
            nodes = set()
            for lbl in labels:
                if lbl not in self._sp_cache:
                    sp = nx.shortest_path(self._g, lbl).keys()
                    self._sp_cache[lbl] = sp
                nodes.update(self._sp_cache[lbl])
            rare = min(labels, key=self._frq.get)
            Xt.append(rare + "-" + "_".join(sorted(nodes)))
        return Xt


def shuffle_arrays(*arrays, **options):
    return [shuffle(a, **options) for a in arrays]


def remove_duplicates(X, idx, stratify=None):
    if stratify is None:
        arr = X[idx]
    else:
        arr = np.asarray([
            "%s_%s" % (xi, si)
            for xi, si in zip(X[idx], stratify[idx])])
    _, unq_idx = np.unique(arr, return_index=True)
    return idx[unq_idx]


def min_count_filter(df, input_col, label_col, min_count):
    indices = np.asarray(range(len(df.index)))
    rare = {}
    if min_count > 1:
        labels = {}
        print("Min count: %d" % min_count)

        for idx, row in df.iterrows():
            for col in label_col:
                if pd.isna(row[col]):
                    raise ValueError("Missing value for %s column "
                                     "for idx=%d" % (col, idx))
                labels.setdefault(col, []).append(row[col].split())
        all_remove_idx = set()
        all_filter_idx = set()
        for col in label_col:
            print("Column: %s" % col)
            cnt = Counter(flatten_list(labels[col]))
            rare[col] = set(
                [cid for cid, c in cnt.items() if c < min_count])

            print("Rare labels: %d" % len(rare[col]))
            # items labelled with all rare
            remove_idx = [idx for idx, lbls in enumerate(labels[col])
                          if len(rare[col].intersection(lbls)) == len(lbls)]

            # items labelled with at least one rare
            filter_idx = [idx for idx, lbls in enumerate(labels[col])
                          if idx not in remove_idx
                          and not rare[col].isdisjoint(lbls)]
            print("Items to be removed: %d (%.2f%%)" % (
                len(remove_idx), 1.0 * len(remove_idx) / len(indices)))
            print(
                "Items to be filtered (train-only): %d (%.2f%%)" % (
                    len(filter_idx), 1.0 * len(filter_idx) / len(indices)))
            all_filter_idx.update(filter_idx)
            all_remove_idx.update(remove_idx)
        all_filter_idx = [idx for idx in all_filter_idx
                          if idx not in all_remove_idx]
        print("Total items to be removed: %d (%.2f%%)" % (
            len(all_remove_idx), 1.0 * len(all_remove_idx) / len(indices)))
        print("Total to be filtered (rare label remove): %d (%.2f%%)" % (
            len(all_filter_idx), 1.0 * len(all_filter_idx) / len(indices)))

        # Remove rows for dataframe
        indices = np.asarray(remove_list_values(indices, all_remove_idx))
        indices = np.asarray(remove_list_values(indices, all_filter_idx))
        df = df.iloc[indices]
        print('-' * 80)
    df, phrases, y = get_filtered_dataset(df, input_col, label_col,
                                          filter_out=rare)
    return df, phrases, y


def get_filtered_dataset(df, input_col, label_col, filter_out=None):
    inputs = []
    outs = []
    for idx, row in df.iterrows():
        lbls = []
        for col in label_col:
            if pd.isna(row[col]):
                raise ValueError("Missing value for %s column "
                                 "for idx=%d" % (col, idx))
            col_lbls = row[col].split()
            if filter_out and filter_out.get(col, None):
                col_lbls = remove_list_values(col_lbls, filter_out[col])
                df.at[idx, col] = " ".join(col_lbls)

            col_lbls = ["%s_%s" % (col, lbl) for lbl in col_lbls]
            lbls.extend(col_lbls)

        outs.append(list(set(lbls)))

        inputs.append(row[input_col].lower())

    outs = np.asarray(outs)
    inputs = np.asarray(inputs)
    return df, inputs, outs


def transfer_seen_instances(X, train_idx, test_idx, max_seen_num):
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    if max_seen_num is None:
        return train_idx, test_idx

    X_test = X[test_idx]
    tcd = dict(Counter(X[train_idx]))
    test_counter = Counter(X_test)
    transfer_idx = []
    for inp, test_cnt in test_counter.most_common():
        train_cnt = tcd.get(inp, 0)
        if train_cnt <= 0:
            continue
        # sr = 1.0 * test_cnt / train_cnt
        if test_cnt <= max_seen_num:
            continue

        # x = (test_cnt - max_seen_ratio * train_cnt /
        #      (1 + max_seen_ratio))
        # d = min(int(round(x)), test_cnt - 1)
        d = test_cnt - max_seen_num
        if d > 0:
            t = test_idx[np.where(X_test == inp)[0][:d]]
            assert len(t) == d
            transfer_idx.extend(t)
    if transfer_idx:
        trans_test_idx = np.in1d(test_idx, transfer_idx).nonzero()[0]
        if len(transfer_idx) != len(trans_test_idx):
            raise ValueError("%d != %d" % (
                len(transfer_idx), len(trans_test_idx)))
        train_idx = np.concatenate((train_idx, transfer_idx))
        test_idx = np.delete(test_idx, trans_test_idx)
    return train_idx, test_idx


class LosslessStratKFold(object):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self._n_splits = n_splits
        self._shuffle = shuffle
        self._random_state = random_state

    def split(self, X, y, max_seen_num=None):
        kf = StratifiedKFold(n_splits=self._n_splits,
                             random_state=self._random_state,
                             shuffle=self._shuffle)
        filter_idx, train_only_idx, stratify_labels = lossless_stratify_filter(
            y, min_count=self._n_splits)

        # filter_idx, train_only_idx, stratify_labels = lossless_stratify_filter(
        #     y, min_count=2)
        flt_stratify = filter_list(stratify_labels, filter_idx)
        filter_idx = np.asarray(filter_idx)
        if len(flt_stratify) == 0:
            raise ValueError("Empty stratification array!")
        for flt_train_idx, flt_test_idx in kf.split(filter_idx, flt_stratify):
            train_idx = filter_idx[flt_train_idx]
            test_idx = filter_idx[flt_test_idx]
            if train_only_idx:
                train_idx = np.concatenate((train_idx, train_only_idx))
            train_idx, test_idx = transfer_seen_instances(
                X, train_idx, test_idx, max_seen_num=max_seen_num)
            if self._shuffle:
                yield shuffle_arrays(train_idx, test_idx,
                                     random_state=self._random_state)
            else:
                yield sorted(train_idx), sorted(test_idx)


class TrainValTestLosslessStratKFold(LosslessStratKFold):
    def __init__(self, n_splits=3, shuffle=False,
                 random_state=None, val_size=None):
        super(TrainValTestLosslessStratKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        if val_size:
            self._val_size = val_size
        else:
            self._val_size = (1.0 / (self._n_splits - 1))

    def split(self, X, y, max_seen_num=None):
        # if max_seen_ratio is not None:
        #     mr = (1.0 - self._val_size) * max_seen_ratio
        # else:
        #     mr = None
        folds = super(TrainValTestLosslessStratKFold, self).split(
            X, y, max_seen_num=max_seen_num)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for train_val_idx, test_idx in folds:
            train_idx, val_idx = lossless_train_test_split(
                train_val_idx, stratify=y[train_val_idx],
                test_size=self._val_size, random_state=self._random_state,
                shuffle=self._shuffle)
            train_idx, val_idx = transfer_seen_instances(
                X, train_idx, val_idx, max_seen_num=max_seen_num)
            # validation
            assert len(train_idx) == len(set(train_idx))
            assert len(val_idx) == len(set(val_idx))
            assert len(test_idx) == len(set(test_idx))

            yield shuffle_arrays(train_idx, val_idx, test_idx,
                                 random_state=self._random_state)


def lossless_stratify_filter(stratify, min_count=2):
    multi_label = (isinstance(stratify[0], list)
                   or isinstance(stratify[0], np.ndarray))
    if multi_label:
        # Important: need to sort multi-labels for each instance for consistency
        stratify_ = [sorted(lbl) for lbl in stratify]
        flatten_list = [l for lbl in stratify_ for l in lbl]
    else:
        stratify_ = stratify
        flatten_list = stratify_

    lbl_counter = Counter(flatten_list).items()
    rare_values = [v for v, cnt in lbl_counter
                   if cnt < min_count]
    is_rare = {v: True for v in rare_values}

    if multi_label:
        rare_idx = [i for i, vals in enumerate(stratify_)
                    if any(is_rare.get(v, False) for v in vals)]
        s = list(stratify_)
        for idx in rare_idx:
            s[idx] = ["REMOVE_%s" % idx]

        mlr = MultiLabelReducer()
        stratify_labels = mlr.fit(s).transform(s)
        strat_counter = Counter(stratify_labels).items()
        rare_stratify = [v for v, cnt in strat_counter
                         if cnt < min_count]

        filter_idx = [i for i, val in enumerate(stratify_labels)
                      if val not in rare_stratify]
    else:
        stratify_labels = stratify
        filter_idx = [i for i, v in enumerate(stratify_)
                      if not is_rare.get(v, False)]
    train_only_idx = [i for i in range(len(stratify_))
                      if i not in filter_idx]
    # print("Filtered size: %d" % len(filter_idx))
    # print("Train only: %d" % len(train_only_idx))
    return filter_idx, train_only_idx, stratify_labels


def lossless_train_test_split(*arrays, **options):
    stratify = options.pop('stratify', None)
    test_size = options.pop('test_size', 0.25)
    if stratify is not None:
        filter_idx, train_only_idx, stratify_labels = lossless_stratify_filter(
            stratify, min_count=2)
        n_classes = len(set(stratify_labels))
        n_test_inst = test_size * len(filter_idx)
        if n_test_inst < 1:
            raise ValueError("Not enough test items!")
        if n_classes > n_test_inst:
            test_size = round(1.0 * n_classes / len(filter_idx), 4)

        flt_arrays = []
        train_only = []
        flt_stratify = filter_list(stratify_labels, filter_idx)
        for a in arrays:
            flt_arrays.append(filter_list(a, filter_idx))
            train_only.append(filter_list(a, train_only_idx))

        result = train_test_split(*flt_arrays,
                                  stratify=flt_stratify,
                                  test_size=test_size,
                                  **options
                                  )
        actual_test_size = None
        for i, a in enumerate(flt_arrays):
            train, test = result[2 * i], result[2 * i + 1]
            train[:0] = train_only[i]  # prepend
            if actual_test_size is None:
                actual_test_size = 1.0 * len(test) / (len(train) + len(test))
        # print("Test size: %.2f | Actual size: %.2f" % (
        #     test_size, actual_test_size))

        if options.get('shuffle', False):
            result = shuffle_arrays(
                *result, random_state=options.get('random_state', None))
        return result
    return train_test_split(*arrays, **options)
