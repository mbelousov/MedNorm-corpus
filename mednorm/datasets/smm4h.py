from mednorm.datasets.base import CsvDatasetConverter


class Smm4hConverter(CsvDatasetConverter):
    def __init__(self, dataset_path, dataset_name=None):
        super(Smm4hConverter, self).__init__(
            dataset_path=dataset_path, dataset_name=dataset_name, sep='\t',
            input_col=1, target_col=2, header=None)

    def print_stats(self, padding=''):
        pass
