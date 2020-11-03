from programs.common_diagnostics import common_diagnostics, numerical_diagnostics, MeanFirstMarginal, LogLT, DiagQuantity, to_string, MeanAllMarginals
import numpy as np
from programs.orthant_proba.model import TemperedOrthantProbability

diag_quantities = common_diagnostics + numerical_diagnostics

def add_to_diag(something):
    diag_quantities.append(something)
    return something

@add_to_diag
class _LogLT(LogLT):
    @staticmethod
    def exact(des):
        return np.nan

@add_to_diag
class _MeanFirstMarginal(MeanFirstMarginal):
    @staticmethod
    def exact(des):
        return np.nan

@add_to_diag
class TimeEvolution(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'time_evolution'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        model: TemperedOrthantProbability = pf.fk.model
        try:
            return to_string(model.list_r)
        except AttributeError:
            return 'N/A'

@add_to_diag
class ThresholdEvolution(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'threshold_evolution'

    @staticmethod
    def extract(pf):
        model: TemperedOrthantProbability = pf.fk.model
        try:
            return to_string(model.list_ar)
        except AttributeError:
            return 'N/A'

    @staticmethod
    def exact(des):
        return None

@add_to_diag
class _MeanAllMarginals(MeanAllMarginals):
    @staticmethod
    def exact(des):
        return np.nan