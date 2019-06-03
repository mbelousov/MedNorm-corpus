import warnings


def import_pymedtermino():
    try:
        import pymedtermino

        pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = False
        pymedtermino.REMOVE_SUPPRESSED_TERMS = False
        pymedtermino.REMOVE_SUPPRESSED_RELATIONS = True
        return pymedtermino
    except ImportError:
        warnings.warn("CANNOT IMPORT PYMEDTERMINO!")
        return None
