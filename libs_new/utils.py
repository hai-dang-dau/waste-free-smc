from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict, namedtuple
from itertools import zip_longest
import multiprocessing as mp
from matplotlib.ticker import MultipleLocator
from particles import smc_samplers as ssp, resampling as rs
import numpy as np
from typing import Iterable, Callable, Tuple, Dict, Sequence, MutableSequence, Any, List, Union
import pandas as pd
import warnings
from numba import NumbaWarning
import scipy.stats as st
from functools import partial
import matplotlib.pyplot as plt
import os
import psutil

class RecyclingParticles(ssp.ThetaParticles):
    """
    Extension of particles.ssp.ThetaParticles with some additional functionalities.
    """

    def __len__(self):
        return len(self.__dict__[self.containers[0]])

    def __repr__(self):
        res = ''
        for i, k in enumerate(self.containers):
            if i > 0:
                res += '\n'
            res += str(k) + ':\n'
            res += str(self.__dict__[k])
        return res

    def copyfrom(self, src, where):
        for k in self.containers:
            self.__dict__[k][where] = src.__dict__[k][where]


def union(l):
    """
    :param l: iterable of RecyclingParticles object
    :return: a RecyclingParticles which is a union of objects in `l`
    """
    res = {}
    for r in l:
        for attr in r.containers:
            if res.get(attr) is None:
                res[attr] = []
            res[attr].append(r.__dict__[attr])
    for attr in res.keys():
        res[attr] = np.concatenate(res[attr])
    return RecyclingParticles(**res)


def get_element(ilist, index):
    try:
        return ilist[index]
    except IndexError:
        return 0.0


def get_element_bis(ilist, index):
    if index < 0:
        return 0
    return ilist[index]


def pre_extractor(pf):
    """
    :param pf: a SMCNew object
    :return: particles (without logliks) and weights
    """
    try:
        X = pf.X.points
    except AttributeError:
        X = pf.X.theta['mu']
    W = pf.wgts.W
    return X, W


class OWA:
    """
    Create rapidly objects with attributes
    """

    # tested
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self):
        return 'OWA:' + str(self.__dict__)


class Iterator:
    """
    An iterator that allows accessing its current state.
    """

    def __init__(self, iterable: Iterable):
        self.it = iter(iterable)
        self.current = None

    def __next__(self):
        self.current = next(self.it)
        return self.current


def find_all_row_indices(x: dict, y: pd.DataFrame):
    """
    Extended version of find_row_index
    """
    res = []
    for i in range(len(y)):
        if subset(x, y.iloc[i, :].to_dict()):
            res.append(i)
    return res


def subset(x: dict, y: dict) -> bool:
    # Tested
    for key in x:
        if x.get(key) != y.get(key):
            return False
    return True


def find_row_index(x: dict, y: pd.DataFrame):
    # Tested
    """
    :return: the first index of a row of y which is a *superset* of x. If there is no such row, return None.
    """
    for i in range(len(y)):
        if subset(x, y.iloc[i, :].to_dict()):
            return i
    return None


def multiply_each_row_with(x, y):
    # tested
    """
    :param x: np array of shape m,n
    :param y: np array of shape n,
    :return: multiply each row of x with y, element-wise
    """
    return (x.T * y).T


def multiply_each_column_with(x, y):
    return x * y


def shut_off_numba_warnings():
    warnings.simplefilter('ignore', category=NumbaWarning)

def shut_off_scipy_warnings():
    warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)

class SelectiveIterator:
    # tested
    """Iterator over a sequence on selected elements. Supports also modification of elements if the underlying sequence structure supports it."""
    def __init__(self, seq: Sequence, select: Sequence[bool]):
        """
        :param seq: sequence to be iterated on
        :param select: a sequence of the same length as `seq` that determines which of its elements are included.
        """
        self._seq = seq
        self._iter = enumerate(zip_with_assert(seq, select))

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # noinspection PyTupleAssignmentBalance
            i, (_, select) = next(self._iter)
            if select:
                return _SequenceItem(seq=self._seq, idx=i)

class _SequenceItem:
    # tested
    """Represents an item of a sequence, with the option to modify it"""
    def __init__(self, seq: Sequence, idx: int):
        self._seq = seq
        self._idx = idx

    @property
    def val(self):
        return self._seq[self._idx]

    @val.setter
    def val(self, value):
        self._seq: MutableSequence
        self._seq[self._idx] = value

selectively_iterate = SelectiveIterator

class SequenceIterator:
    # tested
    """
    Iterates over a sequence, with the option to modify it.
    """
    def __init__(self, seq:Sequence):
        self.seq = seq
        self._iter = iter(range(len(seq)))

    def __iter__(self):
        return self

    def __next__(self):
        i = next(self._iter)
        return _SequenceItem(self.seq, i)

class FunctionWithFixedArguments:
    # tested
    """
    This class creates a new function from another function with some of its arguments fixed. It helps bypass the `lambda` construction which may not be pickable in parallel computing.
    """

    def __init__(self, func: Callable, fixed_positional_arguments: Dict[int, Any]=None, fixed_keyword_arguments: Dict[str, Any]=None):
        self.func = func
        if fixed_positional_arguments is None:
            fixed_positional_arguments = dict()
        if fixed_keyword_arguments is None:
            fixed_keyword_arguments = dict()
        self.fixed_positional_arugments, self.fixed_keyword_arugments = fixed_positional_arguments, fixed_keyword_arguments

    def __call__(self, *args, **kwargs):
        final_args = [None] * (len(args) + len(self.fixed_positional_arugments))
        for position, value in self.fixed_positional_arugments.items():
            final_args[position] = value
        missing_arguments = [i not in self.fixed_positional_arugments for i in range(len(final_args))]
        for input_arg, missing_final_arg in zip_with_assert(args, selectively_iterate(final_args, missing_arguments)):
            missing_final_arg.val = input_arg
        final_kwargs = self.fixed_keyword_arugments.copy()
        final_kwargs.update(kwargs)
        return self.func(*final_args, **final_kwargs)

function_with_fixed_arguments = FunctionWithFixedArguments

# noinspection DuplicatedCode
def maximal_overlap(low: np.ndarray, high: np.ndarray):
    """
    Returns a point belonging to the highest possible number of intervals and that number.
    """
    # tested
    # noinspection PyUnresolvedReferences
    All = np.r_[low, high]
    increments = np.array([1] * len(low) + [-1] * len(high))
    argsort = np.argsort(All)
    counter = np.cumsum(increments[argsort])
    All_sorted = All[argsort]
    max_id = np.argmax(counter)
    return (All_sorted[max_id] + All_sorted[max_id + 1]) / 2, counter[max_id]


class ConfidenceIntervalTool:
    """
    Convenient class to work with Confidence intervals.
    """

    def __init__(self, vals, exact, var_ests=None, sd_ests=None, conf=0.95):
        """
        :param vals: values of the esimates from independent runs
        :param exact: exact value if known, `np.nan` otherwise
        :param var_ests: variance estimates, must be of the same length as `vals`
        :param sd_ests: standard deviation estimates
        """
        self.vals = np.array(vals)
        if exact is None:
            raise ValueError('Use np.nan to represent unknown exact value.')
        self.exact = np.array(exact)
        if ((var_ests is None) + (sd_ests is None)) % 2 == 0:
            raise ValueError('Either variance estimates or sd estimates, but not both, must be provided.')
        self._var_ests, self._sd_ests = np.array(var_ests), np.array(sd_ests)
        self.conf = conf

    @property
    def var_ests(self):
        if self._var_ests is None:
            self._var_ests = self._sd_ests ** 2
        return self._var_ests

    @property
    def sd_ests(self):
        if self._sd_ests is None:
            self._sd_ests = self._var_ests ** 0.5
        return self._sd_ests

    @property
    def _multiplicator(self):
        return st.norm.ppf((1 + self.conf) / 2)

    @property
    def vals_high(self):
        return self.vals + self._multiplicator * self.sd_ests

    @property
    def vals_low(self):
        return self.vals - self._multiplicator * self.sd_ests

    @property
    def nb_in(self):
        if not np.isnan(self.exact):
            return np.sum((self.vals_low < self.exact) * (self.exact < self.vals_high))
        else:
            return maximal_overlap(self.vals_low, self.vals_high)[1]

    def prop_in(self, ci=False):
        if not ci:
            return self.nb_in / len(self.vals)
        else:
            no_in = self.nb_in
            no_out = len(self.vals) - no_in
            return st.beta.ppf([0.025, 0.975], a=no_in + 0.5, b=no_out + 0.5)

    @property
    def log_object(self):
        """
        :return: a new object to work with the logarithm of `self.vals` instead of with `self.vals` themselves.
        """
        return self.__class__(vals=np.log(self.vals), exact=np.log(self.exact),
                              var_ests=self.var_ests / (self.vals ** 2), conf=self.conf)


def assert_pf_same(pf1, pf2):
    assert np.allclose(pf1.X, pf2.X)
    assert np.allclose(pf1.logLt, pf2.logLt)
    assert np.allclose(np.array(pf1.loglt), np.array(pf2.loglt))
    assert np.allclose(np.array(pf1.fk.resampled), np.array(pf2.fk.resampled))
    assert np.allclose(np.array(pf1.fk.ESSs), np.array(pf2.fk.ESSs))
    assert np.allclose(np.array(pf1.fk.chosen_Ns), np.array(pf2.fk.chosen_Ns))
    var_ests = [
        np.allclose(np.array(pf1.fk.varlogLts[m]), np.array(pf2.fk.varlogLts[m]))
        for m in ['init_seq', 'naive', 'th']
    ]
    assert np.allclose(var_ests, np.array([True, True, True]))
    return


class ManuallyCachedFunction:
    # eye-tested
    def __init__(self, f: Callable):
        self.f = f
        # noinspection PyTypeChecker
        self.cached_argument: Tuple[tuple, dict] = (None, None)
        self.cached_result = None
        self.N_cache_access = 0

    def __call__(self, *args, **kwargs):
        if is_same_tuple(args, self.cached_argument[0]) and is_same_dict(kwargs, self.cached_argument[1]):
            self.N_cache_access += 1
            return self.cached_result
        return self.f(*args, **kwargs)

    def cache(self, *args, **kwargs):
        result = kwargs.pop('result')
        self.cached_argument = args, kwargs
        self.cached_result = result


manually_cached_function = ManuallyCachedFunction


def is_same_tuple(t1: tuple, t2: tuple):
    # tested
    if t1 is None or t2 is None:
        return False
    if len(t1) != len(t2):
        return False
    for item1, item2 in zip(t1, t2):
        if not _int_elligent_is(item1, item2):
            return False
    return True


def is_same_dict(d1: dict, d2: dict):
    # tested
    if d1 is None or d2 is None:
        return False
    if len(d1) != len(d2):
        return False
    for k in d1.keys():
        try:
            if not _int_elligent_is(d1[k], d2[k]):
                return False
        except KeyError:
            return False
    return True


def _int_elligent_is(x, y):
    if isinstance(x, int) and isinstance(y, int) and x == y:
        return True
    return x is y


class ManuallyCachedMethod:
    # tested
    def __init__(self, org_func):
        self.org_func = org_func  # Originally defined function (including the `self` argument).

    def __get__(self, instance, owner):
        if instance is None:
            return self.org_func
        else:
            res = manually_cached_function(partial(self.org_func, instance))
            instance.__dict__[self.org_func.__name__] = res
            return res


manually_cached_method = ManuallyCachedMethod


def ESS_ratio(w: rs.Weights) -> float:
    return w.ESS / len(w.lw)


class CachedProperty:
    # tested
    def __init__(self, prop_func):
        self.prop_func = prop_func

    def __get__(self, instance, owner):
        if instance is None:
            return self.prop_func
        res = self.prop_func(instance)
        instance.__dict__[self.prop_func.__name__] = res
        return res


cached_property = CachedProperty


def log_sum_exp_array(x: np.ndarray, axis: int = None):
    # tested
    """Perform `log_sum_exp` over some axis
    """
    x0 = np.max(x)
    r = x - x0
    # assert np.allclose(np.max(np.exp(r)), 1)
    return np.log(np.sum(np.exp(r), axis=axis)) + x0


def generate_naive_variance_estimators(quantities: np.ndarray, N_run: int):
    # tested
    """ Simulate the naive variance estimate of a quantity by running an algorithm for `N_run` times. These `N_run` times are drawn randomly from `quantities`.
    """
    res = []
    for _ in range(len(quantities)):
        chosen_runs = np.random.choice(len(quantities), N_run, replace=False)
        res.append(np.var(quantities[chosen_runs], ddof=1))
    return res


class TestVarianceEstimatorQuality:
    # tested
    def __init__(self, quantities: np.ndarray, var_ests: List[np.ndarray], method_names: List[str], rescaled=False):
        """
        :param var_ests: a list of which each element contains variance estimators produced using some method.
        :param method_names: must be of the same length as `var_ests`.
        """
        self.quantities, self.var_ests = quantities, var_ests
        self.method_names = method_names
        self.true_sd_low, self.true_sd_high = NormalGammaDistribution.uninformative_prior().get_posterior(quantities).get_CI_for_sd()
        self.true_variance_low, self.true_variance_high = self.true_sd_low**2, self.true_sd_high**2
        self.true_variance = np.var(quantities, ddof=1)
        self.true_sd = np.sqrt(self.true_variance)
        self.rescaled = rescaled

    def plot(self, N_run:int, showfliers=None, annotate_with_rmse=False):
        fig, ax_ = plt.subplots()
        ax_: plt.Axes
        with temporary_numpy_seed(0):
            naive_variance_estimator = generate_naive_variance_estimators(self.quantities, N_run)
        # noinspection PyTypeChecker

        def ax_func(ax, annotate_with_rmse_):
            bp = ax.boxplot(
                [naive_variance_estimator] + self.var_ests,
                labels=['{k} runs'.format(k=N_run)] + self.method_names,
                showfliers=showfliers
                            )
            if annotate_with_rmse_:
                # noinspection PyTypeChecker
                annotate_box_plot(bp, [
                    np.sqrt(np.mean((ve - self.true_variance)**2)) for ve in [naive_variance_estimator] + self.var_ests
                ])
            ax.axhline(y=self.true_variance_low, color='black', label='95% CI for variance', linestyle='dashed')
            ax.axhline(y=self.true_variance_high, color='black', linestyle='dashed')
            if self.rescaled:
                ax.set_ylabel('Rescaled variance')
            ax.legend()

        ax_func(ax_, annotate_with_rmse)
        return fig, lambda _ax: ax_func(_ax, False)

    def rescale(self):
        return self.__class__(quantities=self.quantities/self.true_sd, var_ests=[ve/self.true_variance for ve in self.var_ests], method_names=self.method_names, rescaled=True)

def annotate_box_plot(bp: dict, numbers: List[Union[float, None]]):
    # tested
    """
    :param bp: A dictionary which is returned by the box plot plotting function of matplotlib
    :param numbers: numbers that one wants to annotate the box plots with
    """
    bp = BoxPlotResultWrapper(bp)
    ax: plt.Axes = bp.ax
    ax_y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for box, number in zip_with_assert(bp, numbers):
        if number is not None:
            max_box = max(np.r_[box.box[-1], box.fliers]) + 0.02 * ax_y_range
            ax.annotate('{:.2f}'.format(number), (box.abscissa, max_box))
    ax.set_ylim(ymax=ax.get_ylim()[1] + 0.03 * ax_y_range)

class BoxPlotResultWrapper:
    # tested
    def __init__(self, bp: dict):
        """
        Iterate over box plot result. Each iteration returns an object containing features that are specific to one box only. We assume vertical boxplots.
        The `box` attribute contains the y-positions of five horizontal lines that defined that specific box, from the smallest to the highest.
        :param bp: box plot result, returned by the function matplotlib.pyplot.boxplot
        """
        self.bp = bp
        self.whisker_iterator = iter(bp['whiskers'])
        self.fliers_iterator = iter(bp['fliers'])
        self.median_iterator = iter(bp['medians'])
        self.return_type = namedtuple('_box_of_box_plot', ['abscissa', 'box', 'fliers'])
        self.ax = bp['whiskers'][0].axes

    def __iter__(self):
        return self.__class__(self.bp)

    def __next__(self):
        # Direct extraction
        whisker_low = next(self.whisker_iterator).get_xydata()
        whisker_high = next(self.whisker_iterator).get_xydata()
        median = next(self.median_iterator).get_xydata()
        try:
            fliers = next(self.fliers_iterator).get_ydata()
        except StopIteration:
            fliers = np.array([], dtype=float)
        # Treat
        abscissa = whisker_low[0,0]
        box = np.array([whisker_low[1,1], whisker_low[0,1], median[0,1], whisker_high[0,1], whisker_high[1,1]])
        # Finalize
        return self.return_type(abscissa, box, fliers)

class NormalGammaDistribution:
    # tested
    """Normal-gamma distribution as defined by https://en.wikipedia.org/wiki/Normal-gamma_distribution
    """
    def __init__(self, mu, lbd, alpha, beta):
        self.mu, self.lbd, self.alpha, self.beta = mu, lbd, alpha, beta

    def get_posterior(self, data: np.ndarray) -> 'NormalGammaDistribution':
        n = len(data)
        x_bar = np.mean(data)
        s = np.var(data)
        mu_new = (self.lbd * self.mu + n* x_bar)/(self.lbd + n)
        lbd_new = self.lbd + n
        alpha_new = self.alpha + n/2
        beta_new = self.beta + 1/2 * (n*s + self.lbd*n*(x_bar - self.mu)**2/(self.lbd+n))
        # noinspection PyTypeChecker
        return self.__class__(mu=mu_new, lbd=lbd_new, alpha=alpha_new, beta=beta_new)

    def get_CI_for_precision(self, conf: float=0.95) -> Tuple[float, float]:
        q1, q2 = (1-conf)/2, (1+conf)/2
        return tuple(st.gamma.ppf(q=[q1,q2], a=self.alpha, scale=1/self.beta))

    def get_CI_for_mean(self, conf: float=0.95) -> Tuple[float, float]:
        q1, q2 = (1 - conf) / 2, (1 + conf) / 2
        return tuple(st.t.ppf(q=[q1,q2], df=2*self.alpha, loc=self.mu, scale=np.sqrt(self.beta/(self.lbd*self.alpha))))

    def rvs(self) -> Tuple[float, float]:
        T = st.gamma.rvs(a=self.alpha, scale=1/self.beta)
        X = st.norm.rvs(loc=self.mu, scale=np.sqrt(1/(self.lbd*T)))
        return X,T

    @classmethod
    def uninformative_prior(cls):
        return cls(mu=0, lbd=0, alpha=0, beta=0)

    def get_CI_for_sd(self, conf: float=0.95) -> Tuple[float, float]:
        x, y = self.get_CI_for_precision(conf)
        return y**-0.5, x**-0.5

def multinomial_sampling(W: np.ndarray) -> Tuple:
    # tested
    """Multinomial sampling for multi-dimensional numpy arrays.
    :param W: a numpy array of weights, summing to 1
    :returns: a tuple of indices indicating the chosen element
    """
    W_raveled = np.ravel(W)
    #chosen_raveled_index = np.random.choice(len(W_raveled), p=W_raveled)
    chosen_raveled_index = rs.multinomial_once(W_raveled)
    return np.unravel_index(chosen_raveled_index, W.shape)

def array_resampling(resampling_scheme:str, W:np.ndarray, M:int) -> Tuple:
    # tested
    """Resampling scheme for multidimensional numpy weight arrays.
    :param W: a numpy array of weights, summing to 1.
    :returns: a tuple of numpy arrays indicating the chosen elements
    """
    if M == 1 and resampling_scheme == 'multinomial':
        return multinomial_sampling(W)
    W_raveled = np.ravel(W)
    chosen_ravel_idx = rs.resampling(resampling_scheme, W_raveled, M)
    return np.unravel_index(chosen_ravel_idx, W.shape)

class ZipWithAssert:
    """Like zip, but raises AssertionError if iterables are not of the same length."""
    # tested
    def __init__(self, *iterables: Iterable):
        self.iterators = [iter(iterable) for iterable in iterables]

    def __iter__(self):
        return self

    def __next__(self):
        res = []
        for iterator in self.iterators:
            try:
                res.append(next(iterator))
            except StopIteration:
                pass
        if len(res) == 0:
            raise StopIteration
        elif len(res) == len(self.iterators):
            return tuple(res)
        else:
            raise AssertionError

zip_with_assert = ZipWithAssert

def chi_squared_distance(old_lw: np.ndarray, new_lw: np.ndarray) -> float:
    # tested
    """Estimates the chi-squared distance between two distributions from their discrete representations on the same support.
    """
    assert old_lw.shape == new_lw.shape
    old_W = rs.exp_and_normalise(old_lw)
    with np.errstate(invalid='ignore'):
        logG = new_lw - old_lw
        logG[np.isnan(logG)] = 0
    if np.max(logG) == -np.inf:
        return np.inf
    G_tilde = np.exp(logG - np.max(logG))
    return np.sum(old_W * G_tilde**2)/np.sum(old_W * G_tilde)**2 - 1

def get_ESS_ratio(chi2_distance:float) -> float:
    return 1/(chi2_distance + 1)

def get_chi2_distance(ESSr:float) -> float:
    return 1/ESSr - 1

def dict_to_list(d: Dict[int, Any]) -> list:
    res = [None] * (max(d.keys()) + 1)
    for k, v in d.items():
        res[k] = v
    return res

memory_tracker: Dict[int, float] = defaultdict(float)

def memory_tracker_add(t:int):
    memory_tracker[t] = max(memory_tracker[t], psutil.Process(os.getpid()).memory_info().rss/1024**2)

class FixedSizeDict:
    def __init__(self, size:int):
        if size < 1:
            raise ValueError
        self.size = size
        self._d = OrderedDict()

    def __getitem__(self, item):
        return self._d[item]

    def get(self, item):
        return self._d.get(item)

    def __setitem__(self, key, value):
        try:
            self._d.__getitem__(key)
        except KeyError:
            if len(self._d) >= self.size:
                self._pop()
        self._d[key] = value

    def _pop(self):
        return self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return self._d.__repr__()

def identity_function(x):
    return x

class CachedNumpyFunction(ABC):
    # tested
    """
    Speed up the calculation of an expensive function f such that f([x1,..,xn]) returns [y1,...,yn] where x and y are numpy arrays such that g(x[i]) = y[i] for all i and the results for some indices i are already known from previous calculations.
    """
    def __init__(self, f: Callable[[np.ndarray], np.ndarray]):
        self.f = f
        self.N_cache_access = 0

    def __call__(self, x:np.ndarray):
        ret = []
        uncached_idx = []
        uncached_elem = []
        for i, e in enumerate(x):
            c = self.get_cache(e)
            ret.append(c)
            if c is None:
                uncached_idx.append(i)
                uncached_elem.append(e)
        if len(uncached_idx) > 0:
            uncached_res = self.f(np.array(uncached_elem, dtype=x.dtype))
            for i, e, r in zip_with_assert(uncached_idx, uncached_elem, uncached_res):
                ret[i] = r
                self.put_into_cache(e, r)
        self.N_cache_access += len(ret) - len(uncached_idx)
        return np.array(ret)

    @abstractmethod
    def get_cache(self, e):
        ...

    @abstractmethod
    def put_into_cache(self, e, r):
        ...

class CachedNumpyFunctionByHashing(CachedNumpyFunction):
    # tested
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], wrapper: Callable, size: int=np.inf):
        """
        :param f: a vectorized function that takes a numpy array `x` of any length and returns a numpy array `y` of the same length. By vectorized, we meant that there exists a function g such that g(x[i]) = y[i].
        :param wrapper: callable such that `wrapper(x[i])` is an object with appropriately defined `__hash__` and `__eq__` methods
        :param size: size limit of the cache
        """
        super().__init__(f)
        self.wrapper, self.size = wrapper, size
        self._cache = FixedSizeDict(size=size)

    def get_cache(self, e):
        return self._cache.get(self.wrapper(e))

    def put_into_cache(self, e, r):
        self._cache[self.wrapper(e)] = r

    def clear_cache(self):
        self._cache = FixedSizeDict(size=self.size)

cached_numpy_function_by_hasing = CachedNumpyFunctionByHashing

class CachedNumpyFunctionByAttributeInsertion(CachedNumpyFunction):
    # tested
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], inserted_att_name:str):
        super().__init__(f)
        self.inserted_att_name = inserted_att_name

    def get_cache(self, e):
        return e.__dict__.get(self.inserted_att_name)

    def put_into_cache(self, e, r):
        e.__dict__[self.inserted_att_name] = r

cached_numpy_function_by_attribute_insertion = CachedNumpyFunctionByAttributeInsertion

class NumpyHashWrapper:
    def __init__(self, arr:np.ndarray):
        self._arr = arr
        self._hash = hash(arr.tobytes())

    def __eq__(self, other):
        if isinstance(other, NumpyHashWrapper):
            return np.array_equal(self._arr, other._arr)
        else:
            return False

    def __hash__(self):
        return self._hash

def log(x):
    """
    Like np.log, but does not raise errors when x = 0
    """
    with np.errstate(divide='ignore'):
        return np.log(x)

def zoom_plot(ax_func: Callable[[plt.Axes], Any], box_position: Tuple[float, float, float, float], xlim: Tuple[float, float], ylim: Tuple[float, float], fig=None) -> plt.Figure:
    """
    Create a plot using `ax_func` that includes zooming of a region defined by `xlim` and `ylim` and via a box positioned at `box_position`
    :param fig: original figure to zoom on. If `None`, a new figure will be created.
    :param box_position: lower left corner coordinates + height + width, all defined relative to the original figure
    """
    if fig is None:
        fig, ax = plt.subplots()
        ax_func(ax)
    fig: plt.Figure
    ax_inner = fig.add_axes(box_position, label=str(np.random.randint(np.iinfo(np.int32).max)))
    ax_func(ax_inner)
    ax_inner: plt.Axes
    ax_inner.set_xlim(*xlim)
    ax_inner.set_ylim(*ylim)
    try:
        ax_inner.get_legend().remove()
    except AttributeError:
        pass
    ax_inner.set_xlabel('')
    ax_inner.set_ylabel('')
    return fig

global_debugger = dict()

class TemporaryNumpySeed:
    # tested
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # noinspection PyTypeChecker
        np.random.set_state(self.old_state)

temporary_numpy_seed = TemporaryNumpySeed

def set_axis_integer(ax_: plt.Axes, axis='x'):
    if axis=='x':
        axis = ax_.get_xaxis()
    elif axis=='y':
        axis = ax_.get_yaxis()
    else:
        raise ValueError
    axis.set_major_locator(MultipleLocator(1))

def pipe(x, *args):
    """
    Pipe operator, like in R.
    Usage: pipe([1,2,3,4], np.var, np.sqrt) instead of np.sqrt(np.var([1,2,3,4]))
    """
    for f in args:
        x = f(x)
    return x

def naive_sample_without_replacement_via_rejection(N, k):
    # tested
    if k > N:
        raise ValueError
    res = OrderedDict()
    for _ in range(k):
        while True:
            proposal = np.random.randint(low=0, high=N)
            if proposal not in res:
                break
        res.update({proposal: 0})
    return np.array(list(res))

class ParallelWrapper:
    def __init__(self, f):
        self.f = f

    def __call__(self, x, queue, order: int, seed: int):
        np.random.seed(seed)
        result = self.f(x)
        queue.put((order, result))

def parallelly_evaluate(f, parargs:list, n_cores:int, start_method:str) -> list:
    # tested
    ctx = mp.get_context(start_method)
    queue = ctx.Queue()

    wrapped_f = ParallelWrapper(f)
    seeds = naive_sample_without_replacement_via_rejection(2**32, len(parargs))
    processes = [ctx.Process(target=wrapped_f, args=(arg, queue, i, seed)) for (i, arg), seed in zip_with_assert(enumerate(parargs), seeds)]

    # for process in processes:
    #     process.start()
    # res = [queue.get() for _ in parargs]
    # for process in processes:
    #     process.join()
    res = _parallelly_evaluate(processes, queue, n_cores)

    res = {i:r for i, r in res}
    return [res[k] for k in sorted(res)]

def _parallelly_evaluate(processes: List[mp.Process], queue: mp.Queue, n_cores:int) -> list:
    res = []

    def _get_one_result_and_append_to_res():
        incoming_result = queue.get()
        underlying_process = processes[incoming_result[0]]
        underlying_process.join(timeout=10)
        assert underlying_process.exitcode == 0
        res.append(incoming_result)

    for core, process in zip_longest(range(n_cores), processes):
        if (core is not None) and (process is not None):
            process.start()
        elif core is None:
            # there is no more core to execute processes, so we must wait for one process to finish before executing a new one
            _get_one_result_and_append_to_res()
            process.start()
    while len(res) < len(processes):
        _get_one_result_and_append_to_res()

    return res