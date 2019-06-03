import codecs

import numpy as np


class FileReader(object):
    def __init__(self, filepath, encoding=None):
        self._filepath = filepath
        self._encoding = encoding

    def get_encoding(self):
        if self._encoding is None:
            return 'utf-8'
        return self._encoding


class TabularFileReader(FileReader):
    def __init__(self, filepath, encoding=None,
                 sep='\t', input_col=0, label_col=1):
        super(TabularFileReader, self).__init__(filepath, encoding)
        self._sep = sep
        self._input_col = input_col
        self._label_col = label_col

        if not (isinstance(self._label_col, tuple)
                or isinstance(self._label_col, list)):
            self._label_col = [self._label_col]

        self._multi_target = len(self._label_col) > 1

    def read_rows(self):
        with codecs.open(self._filepath, 'r', self.get_encoding()) as fp:
            for line in fp:
                try:
                    yield self.process_line(line)
                except Exception:
                    print("Skip line: %s" % line.strip())
                    continue

    def convert_label(self, txt):
        return txt.strip()

    def process_line(self, line):
        parts = line.split(self._sep)
        inp = parts[self._input_col].strip()
        out = [self.convert_label(parts[c].strip()) for c in self._label_col]
        if len(out) == 1:
            return [inp, out[0]]
        return [inp, out]

    def read_instances(self):
        inputs, labels = list(zip(*self.read_rows()))
        if self._multi_target:
            y = [np.asarray(d) for d in zip(*labels)]
        else:
            y = np.asarray(labels)
        return np.asarray(inputs), y
