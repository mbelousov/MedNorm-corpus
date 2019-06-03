import codecs
import glob
import os

from mednorm.datasets.base import DatasetConverter


class TwAdrLConverter(DatasetConverter):
    def __init__(self, dataset_path, dataset_name=None):
        super(TwAdrLConverter, self).__init__(dataset_path=dataset_path,
                                              dataset_name=dataset_name)

    def convert_lines(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "Invalid dataset path %s!" % self.dataset_path)
        lines = []
        files = glob.glob(os.path.join(self.dataset_path, '*.fold-0.*.txt'))

        for fpath in files:
            with codecs.open(fpath, 'r', 'utf-8') as fp:
                for line_num, line in enumerate(fp):
                    item_id = os.path.basename(fpath) + "_" + str(
                        line_num + 1)
                    umls_cui, _, phrase = line.strip().split("\t")

                    lines.append(self.construct_line(
                        item_id, phrase, umls_cui=umls_cui))
        return lines

    def print_stats(self, padding=''):
        pass
