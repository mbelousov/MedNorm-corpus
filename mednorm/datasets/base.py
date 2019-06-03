import os

import pandas as pd


class DatasetConverter(object):
    def __init__(self, dataset_path, dataset_name=None):
        self.dataset_path = dataset_path
        if dataset_name is None:
            self.dataset_name = os.path.basename(dataset_path).upper()
        else:
            self.dataset_name = dataset_name

    def convert_lines(self):
        raise NotImplementedError("not implemented.")

    def construct_line(self, item_id, phrase,
                       meddra_code='', sct_id='', umls_cui=''):
        return (self.dataset_name, item_id, phrase,
                meddra_code, sct_id, umls_cui)

    def _padded_print(self, message, padding):
        print("%s%s" % (padding, message))

    def print_stats(self, padding=''):
        print("")


class CsvDatasetConverter(DatasetConverter):
    def __init__(self, dataset_path, dataset_name=None,
                 item_id_col=0, input_col=1, target_col=2,
                 target_term='meddra_code',
                 sep=',', **read_kwargs):
        super(CsvDatasetConverter, self).__init__(dataset_path, dataset_name)
        self.input_col = input_col
        self.item_id_col = item_id_col
        self.target_col = target_col
        self.target_term = target_term
        self.sep = sep
        self.read_kwargs = read_kwargs

    def convert_lines(self):
        df = pd.read_csv(self.dataset_path, sep=self.sep, dtype='str',
                         **self.read_kwargs)
        lines = []
        if self.item_id_col is None:
            collection = zip(df.index,
                             df[df.columns[self.input_col]].values,
                             df[df.columns[self.target_col]].values)
        else:
            collection = zip(df[df.columns[self.item_id_col]].values,
                             df[df.columns[self.input_col]].values,
                             df[df.columns[self.target_col]].values)

        for item_id, phrase, target in collection:
            line_params = {'item_id': item_id, 'phrase': phrase,
                           self.target_term: target}
            lines.append(self.construct_line(**line_params))
        return lines
