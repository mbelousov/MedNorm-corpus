import codecs
import glob
import os

from mednorm.datasets.base import DatasetConverter


class AskAPatientConverter(DatasetConverter):
    def __init__(self, dataset_path, dataset_name=None):
        super(AskAPatientConverter, self).__init__(dataset_path=dataset_path,
                                                   dataset_name=dataset_name)

    def convert_lines(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "Invalid dataset path %s!" % self.dataset_path)
        lines = []
        files = glob.glob(os.path.join(self.dataset_path, '*.fold-0.*.txt'))

        for fpath in files:
            with codecs.open(fpath, 'r', 'iso-8859-1') as fp:
                for line_num, line in enumerate(fp):
                    item_id = os.path.basename(fpath) + "_" + str(
                        line_num + 1)
                    cid, _, phrase = line.strip().split("\t")
                    if len(cid) > 12:
                        # skip AMT concepts
                        continue

                    lines.append(self.construct_line(
                        item_id, phrase, sct_id=cid))
        return lines

    def print_stats(self, padding=''):
        pass
