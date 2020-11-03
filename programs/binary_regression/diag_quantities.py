from programs.common_diagnostics import common_diagnostics, numerical_diagnostics, LogLT, MeanFirstMarginal, MeanAllMarginals
import numpy as np

diag_quantities = []
diag_quantities.extend(common_diagnostics + numerical_diagnostics)

def add_to_dq(something):
    diag_quantities.append(something)
    return something

@add_to_dq
class _LogLT(LogLT):
    @staticmethod
    def exact(des):
        return np.nan

@add_to_dq
class _MeanFirstMarginal(MeanFirstMarginal):
    @staticmethod
    def exact(des):
        return np.nan

@add_to_dq
class _MeanAllMarginals(MeanAllMarginals):
    @staticmethod
    def exact(des):
        return np.nan