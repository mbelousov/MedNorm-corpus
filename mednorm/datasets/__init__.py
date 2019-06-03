from mednorm.datasets.cadec import CadecConverter
from mednorm.datasets.smm4h import Smm4hConverter
from mednorm.datasets.tac import TacConverter
from mednorm.datasets.twadrl import TwAdrLConverter
from mednorm.datasets.askapatient import AskAPatientConverter
from mednorm.datasets.twimed import TwiMedConverter


def create_converter(dataset_name, dataset_path):
    cls = {
        'CADEC': CadecConverter,
        'TwADR-L': TwAdrLConverter,
        # 'AskAPatient-folds': AskAPatientConverter,
        'TwiMed-PubMed': TwiMedConverter,
        'TwiMed-Twitter': TwiMedConverter,
        'SMM4H2017-train': Smm4hConverter,
        'SMM4H2017-test': Smm4hConverter,
        'TAC2017_ADR': TacConverter,
    }.get(dataset_name, None)
    if cls is None:
        raise ValueError("Invalid dataset name %s!" % dataset_name)
    return cls(dataset_path=dataset_path, dataset_name=dataset_name)
