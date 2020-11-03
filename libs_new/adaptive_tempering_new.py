import numpy as np
from particles import smc_samplers as ssp
from libs_new.cores_new import SMCNew
from libs_new.smc_samplers_new import SMCSamplerModel
import typing as tp
from libs_new.mcmc_new import MCMC
from libs_new import utils as ut
from scipy.optimize import brentq, minimize_scalar

class AdaptiveTempering(SMCSamplerModel):
    # tested
    def __init__(self, static_model: ssp.StaticModel, mcmc_engine: tp.Type[MCMC], chi_sq_dist:float=1, caching_mode=None, max_cache_size:int=np.inf, hash_and_eq_wrapper: tp.Callable=None):
        """
        :param chi_sq_dist: the desired chi-squared distance between two successive distributions.
        :param caching_mode: the mechanism to avoid calculating the log-likelihood of a particle twice. Available options are `attribute_insertion` and `hash_table`. Options `max_cache_size` and `hash_and_eq_wrapper` are only relevant if `hash_table` is selected.
        """
        super().__init__(mcmc_engine)
        self.static_model = static_model
        self.chi_sq_dist = chi_sq_dist
        self.caching_mode = caching_mode
        if caching_mode == 'hash_table':
            self.wrapped_loglik = ut.cached_numpy_function_by_hasing(f=self.static_model.loglik, wrapper=hash_and_eq_wrapper, size=max_cache_size)
        elif caching_mode == 'attribute_insertion':
            self.wrapped_loglik = ut.cached_numpy_function_by_attribute_insertion(f=self.static_model.loglik, inserted_att_name='_loglik_cache_')
        elif (caching_mode is None) or (caching_mode == 'None') or np.isnan(caching_mode):
            self.wrapped_loglik = self.static_model.loglik
        else:
            raise ValueError
        # Mutable attributes
        self.tempering_exponents: tp.Dict[int, float] = dict() # G_t(x) \propto prior(x) * likelihood(x)^{tempering_exp(t)}
        self.tempering_exponents[-1] = 0

    # noinspection PyUnusedLocal
    def loglik(self, t:int, x: np.ndarray) -> np.ndarray:
        """
        Helper function to avoid calculating loglik of a particle twice.
        """
        # t is here only for historical reasons.
        return self.wrapped_loglik(x)

    def M0(self, N: int) -> np.ndarray:
        return self.static_model.prior.rvs(size=N)

    def logG(self, t: int, x: np.ndarray) -> np.ndarray:
        assert self.tempering_exponents.get(t) is None
        assert x is self.pf_debug_access.X
        loglik_x = self.loglik(t, x)
        self.tempering_exponents[t] = self._get_next_exponent(current_lw=self.pf_debug_access.wgts.lw, loglik=loglik_x, current_exp=self.tempering_exponents[t-1], chi_sq_dist = self.chi_sq_dist)
        if self.pf_debug_access.verbose:
            print('Current tempering exponent: ' + str(self.tempering_exponents[t]))
        return loglik_x * (self.tempering_exponents[t] - self.tempering_exponents[t-1])

    @classmethod
    def _get_next_exponent(cls, current_lw: np.ndarray, loglik: np.ndarray, current_exp:float, chi_sq_dist: float) -> float:
        # should be tested by looking at sampler log
        """Get the next exponent in the tempering problem.
        :param current_lw: Log-weights of the system of particles approximating the current distribution
        :param loglik: the log-likelihood of the aforementioned system of particles
        :param chi_sq_dist: the desired chi-squared distance between the two distributions
        """
        excess_distance = ut.function_with_fixed_arguments(cls._excess_distance, fixed_positional_arguments={0:current_lw, 1:loglik, 2:current_exp, 3:chi_sq_dist})
        if excess_distance(1 - 1e-12) <= 0:
            return 1
        # noinspection PyTypeChecker
        return brentq(excess_distance, a=current_exp + 1e-12, b=1 - 1e-12)

    @classmethod
    def _excess_distance(cls, current_lw: np.ndarray, loglik:np.ndarray, current_exp: float, chi_sq_dist: float, new_exp:float) -> float:
        # function needed to avoid using lambda notation (pickling problem)
        return ut.chi_squared_distance(old_lw=current_lw, new_lw=current_lw + loglik*(new_exp-current_exp)) - chi_sq_dist

    def mcmc_info(self, t: int, x: np.ndarray, req: str):
        if req == 'adapt':
            return True
        if req == 'sigma':
            raise AssertionError
        if req == 'uld':
            return self.static_model.prior.logpdf(x) + self.loglik(t, x) * self.tempering_exponents[t-1]

    def extend(self, t: int, xp: np.ndarray) -> np.ndarray:
        return xp

    def done(self, smc: SMCNew) -> bool:
        return self.tempering_exponents[smc.t - 1] == 1

    def diag_function_for_adaptive_MCMC(self, t:int, x:np.ndarray) -> np.ndarray:
        return self.loglik(t, x)

def get_next_distribution_parameter(current_lw: np.ndarray, logG: tp.Callable[[tp.Any], np.ndarray], search_domain: tp.Any, final_param, chi_sq_dist:float, solver: str='1d'):
    # tested
    """
    Get the next distribution parameter in a SMC tempering sampler.
    :param current_lw: current log-weight of the particles
    :param logG: a function such that `logG(p)` returns the logG of the particles in case the next tempering distribution has parameter `p`
    :param search_domain: the domain for searching `p`. For example, if `p` is a parameter between p0 and 1, one can put (p0,1)
    :param final_param: final parameter, will be returned if tempering turns out to be unnecessary
    :return: the `p` which corresponds to the next distribution
    """
    excess_distance = ut.function_with_fixed_arguments(_excess_distance_, fixed_keyword_arguments=dict(current_lw=current_lw, logG=logG, chi_sq_dist=chi_sq_dist))
    if excess_distance(final_param) <= 0:
        return final_param
    elif solver == '1d':
        # return brentq(excess_distance, search_domain[0], search_domain[1]) old version
        return minimize_scalar(
            fun=RootFindToMinimize(excess_distance, excess_distance(search_domain[0])),
            bounds=search_domain,
            method='bounded'
        ).x
    else:
        raise ValueError('Unknown solver.')

def _excess_distance_(p, current_lw, logG, chi_sq_dist):
    return ut.chi_squared_distance(current_lw, current_lw + logG(p)) - chi_sq_dist

class RootFindToMinimize:
    """
    Change a root finding problem to a minimization problem, while adding a penalization for a certain value
    """
    def __init__(self, f: tp.Callable, p: float):
        self.f = f
        self.p = p

    def __call__(self, x):
        fx = self.f(x)
        if np.allclose(fx, self.p):
            return 9_999_999_999
        else:
            return np.abs(fx)