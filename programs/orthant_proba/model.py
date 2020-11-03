from typing import Callable, Any
import numpy as np
from particles import resampling as rs
from libs_new.cores_new import SMCNew
from libs_new.smc_samplers_new import SMCSamplerModel
from scipy.stats import norm, truncnorm
from programs.orthant_proba.better_truncnorm import new_truncnorm, BetterTruncNormGen
from libs_new.mcmc_new import MCMC, MCMCWithInvalidData
from libs_new import utils as ut
import typing as tp
from libs_new.adaptive_tempering_new import get_next_distribution_parameter

CPU_UNIT_COST = 10_000

def order_variable(Sigma, a, orderer=np.argmin, debug=False):
    # Tested
    """
    Returns an ordered version of the orthant probability problem {P(X >=a), X ~ N(0, Sigma)}.
    """
    d = len(Sigma)
    obj = _VariableOrdering(mu=np.array([0] * d), Sigma=Sigma.copy(), a=a.copy(), dummy=np.arange(d), orderer=orderer, debug=debug)
    new_order = [i for i in obj]
    if debug:
        print(new_order)
    assert len(set(new_order)) == d
    return Sigma[np.ix_(new_order, new_order)], a[new_order]

class _VariableOrdering:
    # eye-tested
    def __init__(self, mu, Sigma, a, dummy, orderer, debug=False):
        self.mu, self.Sigma, self.a, self.dummy, self.orderer = mu, Sigma, a, dummy, orderer
        self.debug = debug

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.dummy) < 1:
            raise StopIteration
        probas = norm.logsf(self.a, loc=self.mu, scale=np.sqrt(np.diag(self.Sigma)))
        chosen_index = self.orderer(probas)
        condition_value = self.__class__.truncnorm_mean(mean=self.mu[chosen_index], var=self.Sigma[chosen_index, chosen_index], a=self.a[chosen_index])
        [self.__class__.reorder_1d(x, chosen_index) for x in [self.mu, self.a, self.dummy]]
        self.__class__.reorder_2d(self.Sigma, chosen_index)
        self.mu, self.Sigma = self.__class__.condition_on_first_variable(mu=self.mu, Sigma=self.Sigma, val=condition_value)
        if self.debug:
            print(self.mu)
            print(self.Sigma)
        self.a = self.a[1:]
        res = self.dummy[0]
        self.dummy = self.dummy[1:]
        return res

    @staticmethod
    def reorder_1d(x,i):
        x[i], x[0] = x[0], x[i]

    @staticmethod
    def reorder_2d(x, i):
        #tested
        x[i,:], x[0,:] = x[0,:].copy(), x[i,:].copy()
        x[:,i], x[:,0] = x[:,0].copy(), x[:,i].copy()

    @staticmethod
    def condition_on_first_variable(mu, Sigma, val):
        # tested
        """
        :param mu: numpy array of size (d,)
        :param Sigma: numpy array of size (d,d)
        :param val: float
        :return: mu_prime, Sigma_prime: mean and covariance matrix of N(X_{1:d-1} | X_0 = val), where X_{0:d-1} ~ N(mu, Sigma)
        """
        mu_prime = Sigma[0, :] / Sigma[0, 0] * (val - mu[0]) + mu
        Sigma_prime = Sigma - np.outer(Sigma[0, :], Sigma[0, :]) / Sigma[0, 0]
        return mu_prime[1:], Sigma_prime[1:, 1:]

    @staticmethod
    def truncnorm_mean(mean, var, a):
        # tested
        """
        Calculate the mean of N(mean, var) conditionnally on x >=a.
        """
        s = var ** 0.5
        return mean + s * truncnorm.moment(1, a=(a-mean)/s, b=np.inf)

#chosen_truncnorm = adhoc_truncnorm
chosen_truncnorm = new_truncnorm
assert isinstance(new_truncnorm.rvs.__self__, BetterTruncNormGen)

class TruncatedGaussianSimulator:
    def __init__(self, rvs=chosen_truncnorm.rvs):
        self.cpu = 0
        self.rvs = rvs

    def truncated_normal_sim(self, lower=None, upper=None, shape=None, infinity=np.inf):
        if lower is None:
            lower = -infinity
        if upper is None:
            upper = infinity
        implicit_shape = False
        if isinstance(lower, np.ndarray):
            shape = lower.shape
            implicit_shape = True
        elif isinstance(upper, np.ndarray):
            shape = upper.shape
            implicit_shape = True
        if implicit_shape:
            res = self.rvs(a=lower, b=upper)
        else:
            res = self.rvs(a=lower, b=upper, size=np.product(shape))
        if res.shape != shape:
            res = res.reshape(shape)
        assert np.product(shape) > 0
        return res

    def truncated_normal_sim_direct(self, lower, upper):
        if (not hasattr(self.rvs, '__self__')) or (not isinstance(self.rvs.__self__, BetterTruncNormGen)):
            return self.truncated_normal_sim(lower, upper)
        N = len(lower)
        assert lower.shape == (N,) and upper.shape == (N,)
        if np.sum(lower > upper) != 0:
            raise RuntimeError('Truncated normal simulation: max(lower-upper)={} which is greater than 0'.format(np.max(lower-upper)))
        # noinspection PyProtectedMember
        return BetterTruncNormGen._real_rvs(a=lower, b=upper)

    def outer(self, a, b):
        assert len(a) > 1 and len(b) > 1
        self.cpu += len(a) * len(b) / CPU_UNIT_COST
        return np.outer(a, b)

class OrthantProbability(SMCSamplerModel):
    def __init__(self, Gamma: np.ndarray, a: np.ndarray, static_model: TruncatedGaussianSimulator):
        super().__init__(OrthantGibbs)
        assert np.allclose(np.diag(Gamma), np.array([1] * len(Gamma)))
        self.Gamma, self.a = Gamma, a
        self.static_model = static_model # simulator and cpu cost accmulator

    def M0(self, N):
        return self.static_model.truncated_normal_sim(lower=self.a[0], shape=(N, 1))

    def logG(self, t, x):
        Gamma_e = self.Gamma[t][0:t].reshape((t,1))
        x = x[:, 0:t]
        truncations = (-x @ Gamma_e + self.a[t]).reshape((len(x),))
        return norm.logsf(truncations)

    def done(self, smc):
        return smc.t >= len(self.Gamma)

    def extend(self, t, xp: np.ndarray):
        Gamma_e = self.Gamma[t][0:t].reshape((t,1))
        new = self.static_model.truncated_normal_sim(lower=-xp @ Gamma_e + self.a[t]) # shape determined atuomatically
        return np.c_[xp, new]

    def mcmc_info(self, t:int, x: np.ndarray, req: str):
        if req == 'Gamma':
            return self.Gamma[0:t, 0:t]
        elif req == 'a':
            return self.a[0:t]
        elif req == 'static_model':
            return self.static_model
        elif req == 't':
            return t
        else:
            raise ValueError('Unknown request.')

class TemperedOrthantProbability(SMCSamplerModel):
    """
    Warning: never run this SMC Sampler model using Adaptive Wasteless SMC with ESSrmin_outer < 1. The reason is that sometimes so many particles are completely eliminated by the hard-thresholding logG function that the algorithm for choosing the length of MCMC chains automatically no longer converges.
    """
    def __init__(self, Gamma: np.ndarray, a: np.ndarray, static_model: TruncatedGaussianSimulator, chi_sq_distance:float=0.7):
        super().__init__(WrappedOrthantGibbs)
        assert np.allclose(np.diag(Gamma), np.array([1] * len(Gamma)))
        self.Gamma, self.a = Gamma, a
        self.static_model = static_model
        self.chi_sq_distance = chi_sq_distance
        # Mutable attributes
        self.r = -1 # current distribution index (from 0 to t)
        self.a_r: float = -np.inf # current tempering parameter
        self.just_added_dimension = False
        # Mutable attributes (logging only)
        self.list_r = []
        self.list_ar = []

    def M0(self, N):
        self.r = 0
        self.a_r = self.a[0]
        self.just_added_dimension = True
        return self.static_model.truncated_normal_sim(lower=self.a[0], shape=(N, 1))

    def _logG(self, t:int, a_t: float, x:np.ndarray, assert_mode:int=1):
        assert x.shape[1] == t + assert_mode
        Gamma_e = self.Gamma[t][0:t].reshape((t, 1))
        x = x[:, 0:t]
        truncations = (-x @ Gamma_e + a_t).reshape((len(x),))
        return norm.logsf(truncations)

    def _log_indicator(self, a_t:float, t:int, x:np.ndarray) -> np.ndarray:
        # tested
        """
        Calculate log(1_{x_t >= a_t - Gamma_{t,0}x_0 - ... - Gamma_{t,t-1}x_{t-1}})
        """
        assert x.shape[1] == t + 1
        # res = x[:,t] >= a_t - x[:,0:t] @ self.Gamma[t,0:t]
        check_bounds = self._check_bounds(t, x)
        self._check_bounds.cache(t, x, result=check_bounds)
        res = check_bounds >= a_t
        with np.errstate(divide='ignore'):
            return np.log(res, dtype=np.float64)

    @ut.manually_cached_method
    def _check_bounds(self, t:int, x:np.ndarray) -> np.ndarray:
        return x[:, t] + x[:, 0:t] @ self.Gamma[t, 0:t]

    def logG(self, t: int, x: np.ndarray) -> np.ndarray:
        if self.just_added_dimension:
            self.just_added_dimension = False
            return self._logG(self.r, self.a_r, x)
        else:
            assert self.a_r != self.a[self.r]
            assert x is self.pf_debug_access.X
            ar_new = get_next_distribution_parameter(current_lw=self.pf_debug_access.wgts.lw, logG=ut.function_with_fixed_arguments(self._log_indicator, fixed_keyword_arguments=dict(t=self.r,x=x)), search_domain=(self.a_r, self.a[self.r]), final_param=self.a[self.r], chi_sq_dist=self.chi_sq_distance)
            self.a_r = ar_new
            return self._log_indicator(ar_new, self.r, x)

    def done(self, smc: SMCNew) -> bool:
        #print('Finished distribution t={} at {}'.format(self.r, self.a_r))
        self.list_r.append(self.r)
        self.list_ar.append(self.a_r)
        return (self.r == len(self.Gamma) - 1) and (self.a_r == self.a[self.r])

    def _extend(self, t:int, a_t:float, xp: np.ndarray):
        Gamma_e = self.Gamma[t][0:t].reshape((t,1))
        new = self.static_model.truncated_normal_sim(lower=-xp @ Gamma_e + a_t) # shape determined atuomatically
        return np.c_[xp, new]

    def extend(self, t: int, xp: np.ndarray) -> np.ndarray:
        # eye-tested
        assert xp.shape[1] == self.r + 1
        if self.a_r != self.a[self.r]:
            return xp
        else:
            a_rp1 = get_next_distribution_parameter(
                current_lw=self.pf_debug_access.fk.current_post_rejuvenate_w.lw,
                logG=ut.function_with_fixed_arguments(
                    self._logG,
                    {0: self.r + 1, 2: xp, 3: 0}
                ),
                search_domain=(-999, self.a[self.r + 1]),
                final_param=self.a[self.r + 1],
                chi_sq_dist=self.chi_sq_distance
            )
            res = self._extend(self.r + 1, a_rp1, xp)
            self.r = self.r + 1
            self.a_r = a_rp1
            self.just_added_dimension = True
            return res

    def mcmc_info(self, t:int, x: np.ndarray, req: str):
        if req == 'Gamma':
            return self.Gamma[0:self.r + 1, 0:self.r + 1]
        elif req == 'a':
            a = self.a[0:self.r + 1].copy()
            a[-1] = self.a_r
            return a
        elif req == 'static_model':
            return self.static_model
        elif req == 't':
            return t
        else:
            raise ValueError

    def diag_function_for_adaptive_MCMC(self, t:int, x:np.ndarray) -> np.ndarray:
        return self._logG(self.r, self.a_r, x)

class OrthantGibbs(MCMC):
    # tested, and numerical instabilities fixed.
    """
    Gibbs sampler for the truncated standard Gaussian on {Gamma * x >= a}
    """
    @classmethod
    def initialize_from_ssp(cls, x: np.ndarray, w: rs.Weights, info: Callable[[np.ndarray, str], Any]) -> 'MCMC':
        if info(x, 't') <= 1:
            raise ValueError('Please set ESSrmin to some value strictly smaller than 1. Gibbs kernel for orthant probability is not implemented (and not necessary) on dimension 1.')
        return cls(Gamma=info(x, 'Gamma'), a=info(x, 'a'), static_model=info(x, 'static_model'))

    def __init__(self, Gamma: np.ndarray, a: np.ndarray, static_model: TruncatedGaussianSimulator, debug=False, refreshment_rate=1/1000):
        """
        :param Gamma: array of shape (d,d)
        :param a: array of shape (d,)
        """
        self.Gamma, self.a = Gamma, a
        self.static_model = static_model
        self.separate_result = dict() # to contain result of the `separate` function below.
        for i in range(len(Gamma)):
            self.separate_result[i] = self.__class__.separate(self.Gamma[:,[i]])
        self.debug = debug
        self.refreshment_rate = refreshment_rate
        self._conformity_checked = False

    def step(self, x):
        if not self._conformity_checked:
            assert check_conformity(self.Gamma, x, self.a)
            self._conformity_checked = True
        x = self._extended_space(x)
        x = (x[0].copy(order='F'), x[1].copy(), int(x[2]))
        for j in range(len(self.a)):
            x = self._update_coordinate_in_place(x=x, j=j)
        assert x[2] == len(self.Gamma) - 1
        self._extended_space.cache(x[0], result=x)
        return x[0]

    @ut.manually_cached_method
    def _extended_space(self, x: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, int]:
        """
        It is more convenient to express things in term of a Markov chain on an extended space.
        """
        N, d = x.shape
        assert d == len(self.Gamma)
        i = d - 1
        mask = np.array([True]*d); mask[i] = False
        a_dN = np.array([self.a]*N).T; assert a_dN.shape == (d, N)
        res = a_dN - self.Gamma[:, mask] @ x[:, mask].T
        self.static_model.cpu += d*(d-1)*N/CPU_UNIT_COST
        if self.debug:
            print('Space extended.')
        return x, res, i

    def _update_coordinate_in_place(self, x:tp.Tuple[np.ndarray, np.ndarray, int], j:int) -> tp.Tuple[np.ndarray, np.ndarray, int]:
        i = x[2]
        y1 = x[1] + self.static_model.outer(self.Gamma[:, j], x[0][:, j]) - self.static_model.outer(self.Gamma[:, i], x[0][:, i])
        if np.random.rand() < self.refreshment_rate:
            # Calculate y1 using the slow method to guarantee numerical stabilities
            n, d = x[0].shape
            a_dn = np.c_[tuple([self.a] * n)]
            mask = np.array([True] * d)
            mask[j] = False
            y1_slow = a_dn - self.Gamma[:, mask] @ (x[0][:, mask]).T
            if self.debug:
                assert np.allclose(y1, y1_slow)
                print('Stablized.')
            y1 = y1_slow
        y2 = j
        lower, upper = self.__class__.solve_limits(self.Gamma[:, [j]], y1, separate_v=self.separate_result[j])
        x[0][:, j] = self.static_model.truncated_normal_sim_direct(lower=lower, upper=upper) # updated in-place
        return x[0], y1, y2

    def free_memory(self):
        self._extended_space.cache(None, result=None)

    @staticmethod
    def solve_limits(v, M, separate_v=None):
        #tested
        """
        :param v: array of shape (d,1)
        :param M: array of shape (d,m)
        :return: two arrays of shape (m,) giving lower limits and upper limits of the x defined by equation vx >= M
        """
        d, m = M.shape
        assert v.shape == (d,1)
        if separate_v is None:
            g0, s0 = OrthantGibbs.separate(v)
        else:
            g0, s0 = separate_v
        if len(g0) == 0:
            lower = np.zeros(shape=(m,)) - np.inf
        else:
            lower = np.max(M[g0,:]/v[g0,:], axis=0)
        if len(s0) == 0:
            upper = np.zeros(shape=(m,)) + np.inf
        else:
            upper = np.min(M[s0, :]/v[s0,:], axis=0)
        return lower, upper

    @staticmethod
    def separate(v):
        #tested
        """
        :param v: array of shape (d,1)
        :return: two lists: indices where v>0 and indices where v<0
        """
        g0 = []; s0 = []
        for i, r in enumerate(v):
            if r[0] > 1e-13:
                g0.append(i)
            elif r[0] < -1e-13:
                s0.append(i)
        return g0, s0

class WrappedOrthantGibbs(MCMCWithInvalidData):
    def check_conformity(self, x:np.ndarray) -> np.ndarray:
        self.mcmc_object: OrthantGibbs
        Gamma = self.mcmc_object.Gamma
        a = self.mcmc_object.a
        return check_conformity(Gamma, x, a, True)

    @classmethod
    def initialize_from_ssp(cls, x:np.ndarray, w: rs.Weights, info: Callable[[np.ndarray, str], Any]) -> 'MCMC':
        mcmc_object = OrthantGibbs.initialize_from_ssp(x, w, info)
        return cls(mcmc_object)

#====Functions for testing/diagnostic purpose===
def check_conformity(Gamma, x, a, array=False):
    N, d = x.shape
    assert Gamma.shape == (d,d)
    a_dN = np.array([a]*N).T
    if not array:
        return np.min(Gamma @ x.T - a_dN) >= 0
    else:
        return np.min(Gamma @ x.T - a_dN, axis=0) >= 0

def extract_samples(samples, a):
    """
    :param samples: array of shape (n,d)
    :param a: array of shape (d,)
    :return: samples that are greater than or equal to a
    """
    is_in = np.product(samples >= a, axis=1)
    return samples[np.where(is_in == 1), :][0]

def generate_readable_Sigma(d):
    """
    Generate an interpretable Sigma matrix (i.e. all diagonals equal to 1)
    """
    Sigma_root_square = np.random.uniform(low=-1, high=1, size=d * d).reshape((d, d))
    Sigma = Sigma_root_square @ Sigma_root_square.T
    normalizer = np.diag(np.sqrt(1 / np.diag(Sigma)))
    return normalizer @ Sigma @ normalizer.T

def generate_Sigma_and_a(d, low=-1, high=2):
    Sigma = generate_readable_Sigma(d)
    a = np.random.uniform(low=low, high=high, size=d)
    multiplicator = np.diag(np.random.uniform(low=1, high=d, size=d))
    Sigma = multiplicator @ Sigma @ multiplicator
    a = multiplicator @ a
    return Sigma, a

def generate_starting_points(Gamma, a, size):
    eta = np.linalg.solve(Gamma, a+0.1)
    return np.array([eta] * size)

def standardized_Gamma_and_a(Sigma, a):
    Gamma = np.linalg.cholesky(Sigma)
    assert np.sum(np.diag(Gamma) > 0) == len(Gamma)
    multiplicator = np.diag(1/np.diag(Gamma))
    Gamma = multiplicator @ Gamma
    a = multiplicator @ a
    return Gamma, a

def standardized_Sigma_and_a(Sigma, a):
    multiplicator = np.diag(1 / np.diag(Sigma)**0.5)
    return multiplicator @ Sigma @ multiplicator.T, multiplicator @ a

def exact_value_estimator(Gamma, a, N, model: TruncatedGaussianSimulator):
    """
    Estimate the exact orthant probability using GHK
    This function's implementation is not completed
    and has been abandoned.
    """
    assert np.allclose(np.diag(Gamma), [1] * len(Gamma))
    x = model.truncated_normal_sim(lower=a[0], shape=(N, 1))
    assert x== 1
    for i in range(len(Gamma) - 1):
        # Simulated until coordinate i. Now makes coordiante i+1
        raise NotImplementedError