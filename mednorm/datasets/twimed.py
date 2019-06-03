import codecs
import glob
import os
import re

from mednorm.datasets.base import DatasetConverter


class TwiMedConverter(DatasetConverter):
    def __init__(self, dataset_path, dataset_name=None):
        super(TwiMedConverter, self).__init__(dataset_path=dataset_path,
                                              dataset_name=dataset_name)
        self.skipped_lines = 0

    def read_annotations(self, ann_file):
        annotations = {}
        entities = {}
        with codecs.open(ann_file, 'r', 'iso-8859-1') as fp:
            for line in fp:
                line = re.sub('\s{2,}', '\t', line)
                try:
                    parts = line.strip().split('\t')
                    if parts[0].startswith("T"):
                        eid, einfo, etxt = parts
                        entities[eid] = etxt
                        continue
                    elif parts[0].startswith('N'):
                        eid, einfo, etxt = parts
                        _, eid, m = einfo.split()
                        _, cid = m.split(':')
                        if cid.startswith('C'):
                            annotations[eid] = (entities[eid], str(cid))
                        else:
                            self.skipped_lines += 1
                except ValueError as e:
                    print("ERROR: %s" % e)
                    continue
        return annotations

    def convert_lines(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "Invalid dataset path %s!" % self.dataset_path)
        ann_files = glob.glob(os.path.join(self.dataset_path, '*.ann'))
        lines = []
        self.skipped_lines = 0
        for ann_file in ann_files:
            fbase = os.path.basename(ann_file)
            annotations = self.read_annotations(ann_file)
            for k, (phrase, umls_cui) in annotations.items():
                item_id = fbase + '_' + k
                lines.append(self.construct_line(
                    item_id, phrase, umls_cui=umls_cui))
        return lines

    def print_stats(self, padding=''):
        self._padded_print("%d skipped lines" % self.skipped_lines,
                           padding=padding)
