import typing as tp
import numpy as np

from libs_new.adaptive_tempering_new import AdaptiveTempering
from libs_new.smc_samplers_new import WastelessSMC
from libs_new.cores_new import SMCNew
from libs_new import utils as ut
from libs_new.mcmc_new import RWMHv2

common_diagnostics = [] # diagnostics common to all problems
numerical_diagnostics = [] # diagnostics specific to numerical problems (and may not be used on, say, Latin squares)

def add_to_common(something):
    common_diagnostics.append(something)
    return something

def add_to_numerical(something):
    numerical_diagnostics.append(something)
    return something

def first_marginal(x):
    return x[:, 0]

class DiagQuantity:
    """
    Represents a diagnostic quantity associated with an executed particle filter.
    """
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def extract(pf):
        """
        how to extract the desired quantity from a pf
        """
        raise NotImplementedError

    @staticmethod
    def exact(des):
        """
        :param des: Description of an algorithm (can be of any type you want)
        :return: exact value of the quantity (say, if we are in the Gaussian toy problem where the exact value is known) ; None if there is no exact value (ie. the quantity is the CPU time); np.nan if exact value is unknown.
        """
        raise NotImplementedError


# noinspection PyTypeChecker
variance_estimation_methods = {'family': ['mcmc'] * 3 + ['LW'] * 5,
                               'variant': ['naive', 'init_seq', 'th'] + [None, 1, 3, 5, 7]} # we verified thoroughly, changing this line is enough to add/remove methods.
variance_estimation_methods['method_list'] = [family + '_' + str(variant) + '_' for family, variant in ut.zip_with_assert(variance_estimation_methods['family'], variance_estimation_methods['variant'])]

def get_family_and_variant(method: str) -> tp.Tuple[str, tp.Union[int, None, str]]:
    i = variance_estimation_methods['method_list'].index(method)
    return variance_estimation_methods['family'][i], variance_estimation_methods['variant'][i]

class VariancePosteriorMean(DiagQuantity):
    # tested
    def __init__(self, name, phi: tp.Callable[[np.ndarray], np.ndarray], method: str):
        """
        :param name: name of the diagnostic - will be used to recognize that diagnostic in the resulting csv file
        :param phi: a function which takes an np array of particles and return an array of floats
        :param method: the method for estimating the variance. List of supported methods in the `variance_estimation_methods['method_list']` attribute of `common_diagnostics.py`.
        """
        self.name_diag, self.phi = name, phi
        self.method_family, self.method_variant = get_family_and_variant(method)

    def name(self):
        return self.name_diag

    def extract(self, pf):
        fk: WastelessSMC = pf.fk
        if self.method_family == 'mcmc':
            try:
                return fk.compute_variance_QT_via_MCMC(self.phi, self.method_variant)
            except (AttributeError, TypeError):
                return np.nan
        elif self.method_family == 'LW':
            try:
                return fk.compute_variance_QT_via_Lee_and_Whiteley(self.phi, no_last_block=self.method_variant)
            except (AttributeError, TypeError):
                return np.nan
        else:
            raise ValueError('Unknown method.')

    def exact(self, des):
        return None

def get_variance_estimators_for_posterior_mean(name, phi):
    return [VariancePosteriorMean(name=name + '_via_' + method, phi=phi, method=method) for method in variance_estimation_methods['method_list']]

numerical_diagnostics.extend(get_variance_estimators_for_posterior_mean('var_mean_first_marginal', first_marginal))

class VarianceLogLT(DiagQuantity):
    def __init__(self, name, method):
        self.name_diag = name
        self.method_family, self.method_variant = get_family_and_variant(method)

    def name(self):
        return self.name_diag

    def extract(self, pf):
        fk: WastelessSMC = pf.fk
        if self.method_family == 'mcmc':
            try:
                return fk.compute_variance_logLT_via_MCMC(self.method_variant)
            except (AttributeError, TypeError):
                return np.nan
        elif self.method_family == 'LW':
            try:
                return fk.compute_variance_logLT_via_Lee_and_Whiteley(self.method_variant)
            except (AttributeError, TypeError):
                return np.nan
        else:
            raise ValueError('Unknown method.')

    def exact(self, des):
        return None

for method_ in variance_estimation_methods['method_list']:
    add_to_common(VarianceLogLT(name='var_logLT_via_' + method_, method=method_))

@add_to_common
class CPUTime(DiagQuantity):
    # tested
    @staticmethod
    def name():
        return 'cpu'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        try:
            return pf.fk.model.static_model.cpu
        except AttributeError:
            return np.nan

@add_to_common
class NoDists(DiagQuantity):
    #tested
    @staticmethod
    def name():
        return 'N_dists'

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        return len(fk.logging.resampling_mode)

    @staticmethod
    def exact(des):
        return None

@add_to_common
class NoResampledDists(DiagQuantity):
    @staticmethod
    def name():
        return 'N_resampled_dists'

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        return len([i for i in fk.logging.resampling_mode if i not in [None, False, 0, -1]])

    @staticmethod
    def exact(des):
        return None

@add_to_common
class MHAcceptanceRate(DiagQuantity):
    @staticmethod
    def name():
        return 'mcmc_acceptance_rate'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        acceptance_rates = [None] * len(fk.logging.resampling_mode)
        for i in range(len(acceptance_rates)):
            mcmc_obj = fk.model.mcmc_object.get(i+1)
            try:
                acceptance_rates[i] = mcmc_obj.acceptance_rate
            except AttributeError:
                acceptance_rates[i] = None
        return to_string(acceptance_rates)

class LogLT(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'logLT'

    @staticmethod
    def extract(pf: SMCNew):
        return pf.logLt

    @staticmethod
    def exact(des):
        raise NotImplementedError

class MeanFirstMarginal(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'mean_first_marginal'

    @classmethod
    def extract(cls, pf: SMCNew):
        return pf.calculate_QT(cls.func())

    @staticmethod
    def exact(des):
        raise NotImplementedError

    @staticmethod
    def func():
        return first_marginal

@add_to_common
class ResamplingMode(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'resampling_mode'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        return to_string(fk.logging.resampling_mode)

@add_to_common
class InnerESSRatio(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'inner_ess_ratio'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        return to_string(fk.logging.inner_ESS_ratio)

@add_to_common
class OuterESSRatio(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'outer_ess_ratio'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        try:
            return to_string(fk.logging.outer_ESS_ratio)
        except AttributeError:
            return 'N/A'

@add_to_common
class ChosenChainLength(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'chosen_chain_length'

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        try:
            return to_string(fk.logging.chain_length)
        except AttributeError:
            return 'N/A'

    @staticmethod
    def exact(des):
        return None

@add_to_common
class ChosenThinning(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'chosen_thinning'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        try:
            return to_string(fk.logging.thinning)
        except AttributeError:
            return 'N/A'

@add_to_common
class CPURaw(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'cpu_raw'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        return pf.cpu_time

@add_to_common
class MemoryUsage(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'memory_in_MB'

    @staticmethod
    def extract(pf):
        return to_string(ut.dict_to_list(ut.memory_tracker))

    @staticmethod
    def exact(des):
        return None

@add_to_common
class TemperingExponents(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'tempering_exponents'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        try:
            model: AdaptiveTempering = pf.fk.model
            return to_string(ut.dict_to_list(model.tempering_exponents))
        except AttributeError:
            return 'N/A'

@add_to_common
class NoOfSavedLogLikEval(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'n_saved_loglik_evals'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        try:
            model: AdaptiveTempering = pf.fk.model
            return model.wrapped_loglik.N_cache_access
        except AttributeError:
            return 0

@add_to_numerical
class CovarianceMatrixCholeskyStatus(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'cov_matrix_cholesky'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf):
        fk: WastelessSMC = pf.fk
        cholesky_status = [None] * len(fk.logging.resampling_mode)
        for i in range(len(cholesky_status)):
            mcmc: RWMHv2 = fk.model.mcmc_object.get(i+1)
            try:
                cholesky_status[i] = mcmc.Sigma_cholesky_decomposition_succeeded
            except AttributeError:
                cholesky_status[i] = None
        return to_string(cholesky_status)

@add_to_common
class LogLts(DiagQuantity):
    @staticmethod
    def name() -> str:
        return 'log_lts'

    @staticmethod
    def exact(des):
        return None

    @staticmethod
    def extract(pf: SMCNew):
        return to_string(pf.loglt, fm='{:.7f}')

def to_string(l, fm='{:.3f}', sep=' '):
    # tested
    """
    :param l: a list of objects
    :param fm: formatting option for floats
    :param sep: separator
    :return: pretty-printing l
    """
    res = ''
    for x in l:
        if _not_really_a_float(x) or _only_zero(x, fm):
            res += sep + str(x)
        else:
            res += sep + fm.format(x)
    return res

def _not_really_a_float(x):
    try:
        y = float(x)
        if np.isinf(y):
            return True
    except (TypeError, ValueError):
        return True
    else:
        if y - int(y) == 0:
            return True
        else:
            return False

def _only_zero(x, fm):
    x = fm.format(x)
    for c in x:
        if c.isdigit() and c != '0':
            return False
    return True

def to_float(s: str, sep=' '):
    """
    :return: a list of floats
    """
    s = s.split(sep=sep)
    if s[0] in ['', sep]:
        s = s[1:]
    if s[-1] in ['', sep]:
        s = s[:-1]
    s = [_floatize(e) for e in s]
    return np.array(s)

def _floatize(e):
    try:
        return float(e)
    except ValueError:
        if e == 'None':
            return None
        else:
            raise ValueError

def _mean_all_marginals(x):
    return np.mean(x, axis=1)

class MeanAllMarginals(MeanFirstMarginal):
    @staticmethod
    def name() -> str:
        return 'mean_all_marginals'

    @staticmethod
    def exact(des):
        raise NotImplementedError

    @staticmethod
    def func():
        return _mean_all_marginals

numerical_diagnostics.extend(get_variance_estimators_for_posterior_mean('var_mean_all_marginals', _mean_all_marginals))