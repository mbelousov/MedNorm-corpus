import codecs
import glob
import os
import re

from mednorm.datasets.base import DatasetConverter


class CadecConverter(DatasetConverter):
    def __init__(self, dataset_path, dataset_name=None):
        super(CadecConverter, self).__init__(dataset_path=dataset_path,
                                             dataset_name=dataset_name)
        self.skipped_lines = 0
        self.file_pairs = 0

    def read_ann_file(self, ann_file):
        annotations = {}
        n_lines = 0
        with codecs.open(ann_file, 'r', 'iso-8859-1') as fp:
            for line in fp:
                n_lines += 1

                line = re.sub('\s{2,}', '\t', line)
                try:
                    packed = line.strip().split('\t')
                    eid, cinfo, txt = packed
                    raw_label = cinfo.strip().split()[0]

                    if raw_label == 'CONCEPT_LESS':
                        # skip CONCEPT_LESS annotations
                        continue

                    if '/' in raw_label:
                        # for multiple concepts use only the first one
                        raw_label = raw_label.split('/')[0]
                    label = int(raw_label)
                    annotations[eid] = (txt, str(label))
                except ValueError as e:
                    # skip badly formatted lines
                    continue
        self.skipped_lines += n_lines - len(annotations)
        return annotations

    def convert_lines(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "Invalid dataset path %s!" % self.dataset_path)
        self.skipped_lines = 0
        self.file_pairs = 0
        meddra_files = glob.glob(
            os.path.join(self.dataset_path, 'meddra', '*.ann'))
        sct_files = glob.glob(os.path.join(self.dataset_path, 'sct', '*.ann'))
        lines = []

        # iterate through meddra and sct file pairs
        for meddra_ann_file, sct_ann_file in zip(meddra_files, sct_files):
            meddra_base = os.path.basename(meddra_ann_file)
            sct_base = os.path.basename(sct_ann_file)
            if meddra_base != sct_base:
                raise ValueError("meddra file does not correspond to sct file!")
            self.file_pairs += 1
            meddra_annotations = self.read_ann_file(meddra_ann_file)
            sct_annotations = self.read_ann_file(sct_ann_file)
            all_keys = set(meddra_annotations.keys()).union(
                sct_annotations.keys())

            for k in all_keys:
                item_id = sct_base + '_' + k
                m = meddra_annotations.get(k, None)
                s = sct_annotations.get(k, None)
                phrase = ""
                meddra_code = ""
                sct_id = ""
                if m:
                    meddra_code = m[1]
                    phrase = m[0]
                if s:
                    if len(s[1]) > 12:
                        # skip AMT concepts
                        continue
                    sct_id = s[1]
                    phrase = s[0]
                lines.append(self.construct_line(
                    item_id, phrase, meddra_code, sct_id))
        return lines

    def print_stats(self, padding=''):
        self._padded_print("%d file pairs" % self.file_pairs,
                           padding=padding)
        self._padded_print("%d skipped lines" % self.skipped_lines,
                           padding=padding)
