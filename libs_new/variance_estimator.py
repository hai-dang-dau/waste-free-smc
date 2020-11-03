import gc
import numpy as np
from typing import Callable, List
from tqdm import tqdm
from scipy.signal import correlate, choose_conv_method

def MCMC_variance(X: np.ndarray, method: str):
    """
    :param X: a numpy array of shape (P,M) that contains M MCMC chains of lengths P
    :return: estimation of sigma^2 in the CLT for MCMC chains, or, equivalently, M*P times the variance of the estimator produced by the whole array X.
    """
    if method == 'naive':
        return MCMC_variance_naive(X)
    if method == 'init_seq':
        return MCMC_init_seq_v2(X)
    if method == 'th':
        return MCMC_Tukey_Hanning(X)
    raise ValueError('Unknown method.')

def _mean_with_weighted_columns(X: np.ndarray, W: np.ndarray):
    # tested
    """
    :param X: array of shape (P,M)
    :param W: array of shape (M,), summing to 1.
    """
    P, M = X.shape
    W = W/P
    return np.sum(X * W)

def MCMC_variance_weighted(X: np.ndarray, W:np.ndarray, method:str):
    # tested
    """Like `MCMC_variance`, but each column of `X` now has a weight W that sums to 1."""
    P, M = X.shape
    return MCMC_variance(M * W * (X - _mean_with_weighted_columns(X, W)), method)

def MCMC_variance_naive(X):
    P, M = X.shape
    means = np.mean(X, axis=0)
    return np.var(means) * P

def autocovariance(X: np.ndarray, order: int, mu: float = None, bias=True):
    # tested
    if mu is None:
        mu = np.mean(X)
    X = X - mu
    P, M = X.shape
    if bias:
        return np.mean(X[0:(P - order)] * X[order:P]) * (P-order)/P # use the biased estimator
    else:
        return np.mean(X[0:(P - order)] * X[order:P])  # * (P-order)/P # use the biased estimator

def autocovariance_fft_single(x, mu=None, bias=True):
    """
    :param x: numpy array of shape (n,)
    :return: numpy array `res` of shape(n,), where `res[i]` is the i-th order autocorrelation
    """
    # tested
    if mu is None:
        mu = np.mean(x)
    x = x - mu
    res = correlate(x, x, method='fft')
    res = np.array([res[-len(x)+i] for i in range(len(x))])
    if bias:
        return res/len(x)
    else:
        return res/np.arange(len(x),0,-1)

def autocovariance_fft_multiple(X, mu=None, bias=True):
    """
    :param X: numpy array of shape (P,M), which corresponds typically to `M` MCMC runs of length `P` each.
    :return: numpy array `res` of shape (P,), where `res[i]` is the i-th order autocorrelation
    """
    # tested
    if mu is None:
        mu = np.mean(X)
    P, M = X.shape
    res = np.array([autocovariance_fft_single(x=X[:,m], mu=mu, bias=bias) for m in range(M)])
    return np.mean(res, axis=0)

class AutoCovarianceCalculator:
    # tested
    """
    An artificial device to efficiently calculate the autocovariances based on (possibly) multiple runs of an MCMC method.
    """
    def __init__(self, X:np.ndarray, method:str=None, bias=True):
        """
        :param X: np array of size `(M,P)`, typically the result of `M` independent MCMC runs of length `P`
        :param method: how will the covariances be calculated. `None` to let things be chosen automatically, otherwise `direct` or `fft` must be specified.
        """
        self.X = X
        self.P, self.M = X.shape
        # noinspection PyTypeChecker
        self.mu: float = np.mean(X)
        self.method = method
        self.bias = bias
        self._covariances = np.array([np.nan]*self.P)

    def __getitem__(self, k:int):
        if k >= len(self._covariances) or k < 0:
            raise IndexError
        if np.isnan(self._covariances[k]):
            if self.method is None:
                self._choose_method()
            if self.method == 'fft':
                self._covariances = autocovariance_fft_multiple(X=self.X, mu=self.mu, bias=self.bias)
                assert len(self._covariances) == self.P
            elif self.method == 'direct':
                self._covariances[k] = autocovariance(X=self.X, order=k, mu=self.mu, bias=self.bias)
            else:
                raise AssertionError("Method must be either 'fft' or 'direct'")
        return self._covariances[k]

    def _choose_method(self):
        if self.P <= 10:
            self.method = 'direct'
            return
        test = self.X[0:self.P//2,0]
        self.method = choose_conv_method(test, test)

    def __len__(self):
        return len(self._covariances)

class StrictList:
    """
    List with strict indexing rules, used for testing
    """
    def __init__(self, *args, **kwargs):
        self.list = list(*args, **kwargs)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        if item >= len(self) or item < 0:
            raise IndexError
        return self.list[item]

def MCMC_init_seq_v2(X: np.ndarray, method=None, bias=True):
    # tested
    """
    initial sequence estimator, see Practical MCMC (Geyer 1992)
    Let c_0, c_1, ... be the sequence of autocorrelations. Then:
    * i is an inadmissible index if i is odd and one of the two following conditions is proved to be False:
        * c[i] + c[i-1] >= 0
        * c[i-2] + c[i-3] - c[i] - c[i-1] >= 0
    * All c_i are admissible until the first inadmissible index, or when the list runs out.
    """
    covariances = AutoCovarianceCalculator(X=X, method=method, bias=bias)
    i = 0
    while (i< len(covariances)) and (not _inadmissible(covariances, i)):
        i = i + 1
    return -covariances[0] + 2*sum([covariances[j] for j in range(i)])

def _inadmissible(c, i:int):
    # tested
    """Helper for `MCMC_init_seq`
    :param c: an indexable object
    """
    if i % 2 == 0:
        return False
    try:
        val1 = c[i] + c[i-1]
    except IndexError:
        val1 = np.inf
    try:
        val2 = c[i-2] + c[i-3] - c[i] - c[i-1]
    except IndexError:
        val2 = np.inf
    return val1 < -1e-10 or val2 < -1e-10

def MCMC_init_seq(X: np.ndarray):
    """
    initial sequence estimator, see Practical MCMC (Geyer 1992)
    Deprecated, please use `MCMC_init_seq_v2` instead.
    """
    if X.shape[0] == 1:
        return np.var(X)
    mu = np.mean(X)
    autocovariances = []
    sum_of_adjacent_cov = []
    k = -2
    while (k+4) <= X.shape[0]:
        k += 2 # we now calculate autocovariances of order k and k + 1
        # noinspection PyTypeChecker
        autocovariances += [autocovariance(X=X, order = k, mu=mu), autocovariance(X=X, order = k+1, mu=mu)]
        assert len(autocovariances) == k+2
        sum_of_adjacent_cov.append(autocovariances[-1] + autocovariances[-2])
        if (k >= 2) and ((sum_of_adjacent_cov[-1] > sum_of_adjacent_cov[-2] + 1e-12) or (sum_of_adjacent_cov[-1] < -1e-12)):
            break
    #assert np.abs(np.sum(autocovariances) - np.sum(sum_of_adjacent_cov)) < 1e-5
    return -autocovariances[0] + 2 * np.sum(autocovariances)

def MCMC_Tukey_Hanning(X, method=None, bias=True, adapt_constant=True):
    # tested, recommended as default.
    """
    MCMC Variance estimator using spectral variance method with Tukey_Hanning window.
    See `Batch means and spectral variance estimators in MCMC, Flegal and Johns (2010)`
    """
    if np.var(X) < 1e-12:
        return 0
    covariances = AutoCovarianceCalculator(X=X, method=method, bias=bias)
    alpha = 1/4
    P = len(covariances)
    if adapt_constant:
        c = np.sqrt(3.75*MCMC_variance_naive(X)/np.var(X)) # leave this alone for the moment. In high dimensional settings, it is rare that we can run Markov chain for (a lot) more than 3 autocorrelation time.
    else:
        c = 1
    b = max(c * P**0.5+1,2); b = int(b)
    w = [1 - 2*alpha + 2*alpha * np.cos(np.pi*k/b) for k in range(b)]
    w_cov = []
    for i in np.arange(1,b):
        try:
            w_cov.append(w[i] * covariances[i])
        except IndexError:
            w_cov.append(0)
    return w[0] * covariances[0] + 2 * sum(w_cov)

def default_collector(l: List[np.ndarray]) -> np.ndarray:
    gc.collect()
    return np.r_[tuple(l)]

def _weighted_variance_by_columns(x: np.ndarray, W: np.ndarray) -> float:
    # tested
    """Calculate the variance of elements of `x` where each column of `x` is weighted by `W`.
    :param W: weights, should sum to 1
    """
    P, M = x.shape
    W = W/P
    mean_of_squares = np.sum(W * x**2)
    square_of_mean = np.sum(W*x)**2
    return mean_of_squares - square_of_mean

class TargetedReproducer:
    # tested beyond doubts
    """
    Apply repeatedly a specified Markov kernel on a given array of particles until the whole system of particles reaches a desired effective sample size.
    """
    def __init__(self, starting_points: np.ndarray, starting_W: np.ndarray, target_ess: int,
                 kernel: Callable[[np.ndarray], np.ndarray],
                 f: Callable[[np.ndarray], np.ndarray], method:str, union_function: Callable[[List[np.ndarray]], np.ndarray]=None, max_N_particles:float=np.inf, verbose: bool=False, forceful=False, ignore_weights=True):
        """
        :param starting_points: starting system of particles
        :param starting_W: starting weights, should sum to 1
        :param target_ess: desired effective sample size
        :param kernel: Markov kernel
        :param f: function used to calculate ESS
        :param method: method to calculate ESS
        :param forceful: whether to force running Markov chains until var(f) > 0.
        :param max_N_particles: the maximum number of particles.
        :param ignore_weights: ignore starting weights. Recommended to make algorithm more stable.
        """
        if not np.allclose(np.sum(starting_W), 1):
            raise ValueError
        if max_N_particles < 2 * target_ess:
            raise ValueError
        self.starting_points, self.W = starting_points, starting_W
        self.kernel, self.f = kernel, f
        self.target_ess, self.union_function = target_ess, union_function
        self.method = method
        self.verbose, self.forceful = verbose, forceful
        self.max_N_particles = max_N_particles
        if union_function is None:
            self.union_function = default_collector
        self.M = len(self.starting_points)
        if ignore_weights:
            self.W = np.array([1/self.M] * self.M)
        #===All mutuable attributes here===
        self.collect = [self.starting_points]
        self.state = self.starting_points
        self.k = 1 # thinning: pick every k-th chains
        self.kernel_call = 0
        self._ess = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.ess >= self.target_ess:
            raise StopIteration
        if self.current_N_particles * 2 > self.max_N_particles:
            self.k *= 2
            self.collect = list(reversed(self.collect[::-2]))
        if self.verbose:
            print('ESS={ess:.2f},thinning={k},chain length={kc}'.format(ess=self._ess, k=self.k, kc=self.kernel_call+1))
        for _ in tqdm(range(len(self.collect)), disable=not self.verbose):
            for __ in range(self.k):
                self.state = self.kernel(self.state)
                self.kernel_call += 1
            self.collect.append(self.state)

    def run(self) -> np.ndarray:
        for _ in self:
            pass
        return self.union_function(self.collect)

    @property
    def ess(self) -> float:
        mcmc_arr = self.f(self.union_function(self.collect)).reshape(len(self.collect), self.M) # a little wasteful here, but in practice, f should not be an expensive function, so it goes. And in fact caching f may actually be slower in Python...
        sigma_squared_of_distribution = _weighted_variance_by_columns(mcmc_arr, self.W)
        if np.allclose(sigma_squared_of_distribution, 0):
            self._ess = 0 if self.forceful else np.inf
            return self._ess
        sigma_squared_of_MCMC = MCMC_variance_weighted(mcmc_arr, self.W, method=self.method)
        res = sigma_squared_of_distribution/sigma_squared_of_MCMC * self.current_N_particles
        self._ess = res
        return res

    @property
    def current_N_particles(self) -> int:
        return len(self.collect) * self.M