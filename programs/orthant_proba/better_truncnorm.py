# noinspection PyProtectedMember
from scipy.stats._continuous_distns import _norm_cdf, _norm_sf, _norm_isf, _norm_ppf, truncnorm_gen
try:
    # noinspection PyProtectedMember
    from scipy.stats._continuous_distns import _truncnorm_ppf
except ImportError:
    _truncnorm_ppf = None # no problem if you won't be using AdhocTruncNormGen.
import scipy.special as spec
import numpy as np
from scipy.stats import truncexpon
from typing import Callable

class AdhocTruncNormGen(truncnorm_gen):
    """
    This class helps to work around errors in scipy.truncnorm.rvs method (as of scipy version 1.4). It will be no longer necessary once these errors are officially fixed.
    """
    # noinspection PyMethodMayBeStatic
    def _get_norms(self, a, b):
        _nb = _norm_cdf(b)
        _na = _norm_cdf(a)
        _sb = _norm_sf(b)
        _sa = _norm_sf(a)
        _delta = np.where(a > 0, _sa - _sb, _nb - _na)
        return _na, _nb, _sa, _sb, _delta, np.log(_delta)

    def _ppf(self, q, a, b):
        N = len(q)
        if a.shape != ():
            assert len(a) == N
            assert (q.shape == a.shape) and (a.shape == b.shape)
            shape = q.shape
            q = q.reshape((N,))
            a = a.reshape((N,))
            b = b.reshape((N,))
            res = np.zeros(N)
            id_v131 = (a <= 32)
            id_v141 = ~id_v131
            if np.sum(id_v131) > 0:
                res[id_v131] = self._ppf_version_131(q[id_v131], a[id_v131], b[id_v131])
            if np.sum(id_v141) > 0:
                res[id_v141] = self._ppf_version_141(q[id_v141], a[id_v141], b[id_v141])
            return res.reshape(shape)
        else:
            try:
                return self._ppf_version_131(q, a, b)
            except FloatingPointError:
                return self._ppf_version_141(q, a, b)

    def _ppf_version_131(self, q, a, b):
        with np.errstate(divide='raise', over='raise', under='ignore', invalid='raise'):
            ans = self._get_norms(a, b)
            _na, _nb, _sa, _sb = ans[:4]
            ppf = np.where(a > 0,
                           _norm_isf(q * _sb + _sa * (1.0 - q)),
                           _norm_ppf(q * _nb + _na * (1.0 - q)))
            return ppf

    # noinspection PyMethodMayBeStatic
    def _ppf_version_141(self, q, a, b):
        return _truncnorm_ppf(q, a, b)

adhoc_truncnorm = AdhocTruncNormGen(name='adhoc_truncnorm')

def truncnorm_via_rejection(a: np.ndarray, b:np.ndarray, verbose=False):
    # tested
    """
    :param a: numpy array of length n
    :param b: numpy array of length n
    :return: x: numpy array of length n, where x[i] is the truncated normal simulated between a[i] and b[i]
    """
    n = len(a)
    #assert a.shape == (n,) and b.shape == (n,) and np.sum(a>=0) == n and np.sum(b>=0) == n
    lbd = 1/2 * (a + (a**2 + 4)**0.5) # see Robert(1995)
    if verbose:
        print('lbd[0]: ' + str(lbd[0]))
    x_0 = np.minimum(b, lbd)
    logM = -x_0**2/2 + lbd * x_0
    missing = np.array([True] * n)
    x = np.zeros(n)
    i = 0
    while np.any(missing):
        i += 1
        x[missing] = one_accept_reject_step(a=a[missing], b=b[missing], lbd=lbd[missing], logM=logM[missing])
        missing = np.isnan(x)
        if verbose:
            print('Rejection round ' + str(i) + ' ...')
    return x

def one_accept_reject_step(a, b, lbd, logM):
    """
    One step of the accept-reject algorithm for truncated normal simulation between `a` and `b`. This function is helper for `truncnorm_via_rejection`.
    :return: an array `x` of the same length as `a`. The element `x[i]` follows the Gaussian distribution truncated between `a` and `b` if rejection sampling succeeds, otherwise is `np.nan`.
    """
    x = a + 1/lbd * truncated_exponential_2(b=lbd*(b-a))
    log_u = np.log(np.random.rand(len(a)))
    log_p_accept = -x**2/2 + lbd*x - logM
    #assert np.sum(log_p_accept > 1e-10) == 0
    x[log_u > log_p_accept] = np.nan
    return x

def truncated_exponential_1(b):
    return truncexpon.rvs(b=b)

def truncated_exponential_2(b):
    # a little bit faster than `truncated_exponential_1` thanks to no argument checking
    u = np.random.rand(len(b))
    # noinspection PyProtectedMember
    return truncexpon._ppf(u, b)

def truncnorm_via_rejection_negative(a, b):
    """
    :param a,b: negative arrays of size (N,)
    """
    N = len(a)
    #assert a.shape == (N,) and b.shape== (N,) and np.sum(a > 0) == 0 and np.sum(b>0) == 0
    return -truncnorm_via_rejection(-b, -a)


# noinspection PyCallingNonCallable
def truncnorm_via_inverse_cdf(a: np.ndarray, b:np.ndarray):
    # tested
    u = np.random.uniform(low=spec.ndtr(a), high=spec.ndtr(b))
    return spec.ndtri(u)

def truncnorm_via_repeated_norm(a: np.ndarray, b:np.ndarray, verbose=False):
    # tested
    #assert np.sum(a > b) == 0
    n = len(a)
    x = np.zeros(n)
    missing = np.array([True] * n)
    i = 0
    while np.any(missing):
        i+=1
        x[missing] = one_naive_accept_reject(a=a[missing], b=b[missing])
        missing = np.isnan(x)
        if verbose:
            print('Rejection round ' + str(i))
    return x

def one_naive_accept_reject(a, b):
    x = np.random.normal(size=len(a))
    x[np.logical_or(x < a, x > b)] = np.nan
    return x

class BetterTruncNormGen(truncnorm_gen):
    # tested in vivo
    def _rvs(self, *args):
        # eye-tested
        assert len(args) == 2
        a = args[0]; b = args[1]
        size = int(np.prod(self._size))
        if np.prod(a.shape) == 1:
            assert np.prod(b.shape) == 1
            a = self.__class__._multiplier(a, size)
            b = self.__class__._multiplier(b, size)
        else:
            a = a.reshape((size,))
            b = b.reshape((size,))
        res = self.__class__._real_rvs(a, b) # for distribution
        return res.reshape(self._size)

    @staticmethod
    def _multiplier(a:np.ndarray, size:int) -> np.ndarray:
        """
        :param a: a numpy array containing only one element e
        :return [e,e,...,e]
        """
        while True:
            try:
                a = a[0]
            except IndexError:
                break
        return np.array([a] * size)

    @staticmethod
    def _real_rvs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # eye-tested
        """
        :param a,b: numpy arrays of shape (N,)
        :return res: numpy array of shape (N,), where res[i] is a random truncated Gaussian between a[i] and b[i]
        """
        obj = _RealRVS(a=a, b=b)
        obj.fill(eligibility=lambda ap, bp: (np.abs(ap) <=2) * (np.abs(bp) <=2), generator=truncnorm_via_inverse_cdf)
        obj.fill(eligibility=lambda ap,bp: ap*bp <=0, generator=truncnorm_via_repeated_norm)
        obj.fill(eligibility=lambda ap,bp: ap >=0, generator=truncnorm_via_rejection)
        obj.fill(eligibility=lambda ap,bp: ap < 0, generator=truncnorm_via_rejection_negative)
        return obj.get()

class _RealRVS:
    def __init__(self, a, b, verbose=False):
        self.a, self.b = a, b
        self.verbose = verbose
        N = len(a)
        assert a.shape == (N,) and b.shape == (N,)
        self.res = np.array([np.nan] * N)

    def fill(self, eligibility: Callable, generator: Callable):
        with np.errstate(invalid='ignore'):
            eligible = np.isnan(self.res) * eligibility(self.a, self.b)
        self.res[eligible] = generator(self.a[eligible], self.b[eligible])
        if self.verbose:
            print(generator)
            print(np.sum(eligible))

    def get(self):
        #assert np.sum(np.isnan(self.res)) == 0
        return self.res

new_truncnorm = BetterTruncNormGen(name='new_truncnorm')