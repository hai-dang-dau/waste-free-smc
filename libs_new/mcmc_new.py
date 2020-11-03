from typing import Callable, Tuple, Union, Any, List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from tqdm import tqdm
from abc import abstractmethod, ABC
from particles import resampling as rs
from libs_new import utils as ut

class MCMC(ABC):
    """
    Generic Parallel Markov Chain Monte Carlo base abstract class.
    """
    @abstractmethod
    def step(self, x:np.ndarray) -> np.ndarray:
        ...

    @classmethod
    @abstractmethod
    def initialize_from_ssp(cls, x:np.ndarray, w: rs.Weights, info: Callable[[np.ndarray, str], Any]) -> 'MCMC':
        """
        This method initializes an MCMC kernel based on information from an approximate sample `x` with weights `w` from the invariant measure.
        """
        ...

    def run(self, niter: int, start: np.ndarray, verbose, history) -> List[np.ndarray]:
        chain = [start]
        for _ in tqdm(range(niter), disable=not verbose):
            if history:
                chain.append(self.step(chain[-1]))
            else:
                chain = [self.step(chain[-1])]
        return chain

    def free_memory(self):
        """
        Clean up the object once it is no longer actively used.
        """
        pass

class MetropolisHastingsv2(MCMC, ABC):
    # tested
    """
    Basic parallel Metropolis-Hastings.
    """
    def __init__(self, uld: Callable[[np.ndarray], np.ndarray]):
        """
        :param uld: unnormalized log density function, which takes N particles and return a float numpy array of unnormalized log density function for each point.
        """
        self.uld = ut.manually_cached_function(uld)
        self.acceptance_rate, self.n_seen_particles = 0, 0

    @abstractmethod
    def proposal(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the proposal corrections and proposal points.
        correction = logp(x*|x) - logp(x|x*)
        """
        ...

    def step(self, x):
        N = len(x)

        uld_x = self.uld(x)
        correction, xstar = self.proposal(x)
        uld_xstar = self.uld(xstar)

        log_accept_proba = uld_xstar - uld_x - correction
        rejected_positions = np.log(np.random.rand(N)) > log_accept_proba
        xstar[rejected_positions] = x[rejected_positions]
        uld_xstar[rejected_positions] = uld_x[rejected_positions]

        self.uld.cache(xstar, result=uld_xstar)

        n_accepted_particles = N - np.sum(rejected_positions)
        self.acceptance_rate = 1/(self.n_seen_particles + N) * (self.n_seen_particles * self.acceptance_rate + n_accepted_particles)
        self.n_seen_particles += N

        return xstar

    def free_memory(self):
        self.uld.cache(None, result=None)

class RWMHv2(MetropolisHastingsv2):
    # tested
    """
    Parallel Random-walk Metropolis-Hastings.
    """
    @classmethod
    def initialize_from_ssp(cls, x: np.ndarray, w: rs.Weights, info: Callable[[np.ndarray, str], Any]) -> 'MCMC':
        adapt: bool=info(x, 'adapt')
        if adapt:
            Sigma = (2.38 ** 2) / x.shape[1] * np.cov(x, aweights=w.W, rowvar=False)
        else:
            Sigma = info(x, 'sigma') ** 2
        #return cls(uld=lambda _x: info(_x, 'uld'), Sigma=Sigma, d=x.shape[1]) # not pickable!
        # noinspection PyTypeChecker
        return cls(uld=ut.function_with_fixed_arguments(info, {1:'uld'}), Sigma=Sigma, d=x.shape[1])

    def __init__(self, uld: Callable[[np.ndarray], np.ndarray], Sigma: Union[np.ndarray, float], d:int):
        """
        Caution: N samples from a uni-dimensional distribution should always be represented by a numpy array of shape (N,1)
        :param Sigma: covariance matrix of the proposal distribution. If scalar, will be converted to the identity matrix times that scalar.
        """
        super().__init__(uld=uld)
        self.d = d
        if isinstance(Sigma, np.ndarray) and Sigma.shape == (self.d, self.d):
            self.Sigma = Sigma
        else:
            self.Sigma = Sigma * np.identity(self.d)
        try:
            self.Gamma = cholesky(self.Sigma)
            self.Sigma_cholesky_decomposition_succeeded = True
        except LinAlgError:
            self.Gamma = cholesky(np.diag(np.diag(self.Sigma)))
            self.Sigma_cholesky_decomposition_succeeded = False

    def proposal(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.random.normal(size=x.shape)
        y = z @ self.Gamma
        return np.zeros(len(x)), x + y

class MCMCWithInvalidData(MCMC, ABC):
    # tested
    """
    Wrapper for an existing `MCMC` object to deal with objects outside of the intended domain. For example, a RWMH object that simulates a Gaussian distribution will probably return an error message if its `step` function is fed with an array containing both numbers and strings. The objects of this class, however, wraps the aforementioned `step` function so that it leaves alone the strings, while performing desired MCMC operations on the numbers. This class however contains very specific features designed to be used with the `SMCSamplerModel` class.
    """
    def __init__(self, mcmc_object: MCMC):
        self.mcmc_object = mcmc_object
        self.last_x = None
        self.valid_idx = None

    @abstractmethod
    def check_conformity(self, x:np.ndarray) -> np.ndarray:
        """
        Returns a numpy array of booleans specifying whether each item of `x` falls into the intended domain of `self.mcmc_object`.
        """
        ...

    def step(self, x:np.ndarray) -> np.ndarray:
        if self.valid_idx is None:
            self.valid_idx = self.check_conformity(x)
        if self.last_x is not None:
            assert x is self.last_x
        x = x.copy()
        x[self.valid_idx] = self.mcmc_object.step(x[self.valid_idx])
        self.last_x = x
        return x

    def free_memory(self):
        self.last_x = None
        self.valid_idx = None
        return self.mcmc_object.free_memory()

    def __getattr__(self, item):
        return getattr(self.mcmc_object, item)

def vertical_convergence_plot(chain, f, exact_low, exact_high):
    """
    :param chain: chain[t] is the system of particles at time t in parallel MCMC
    :param f: diagnostic function that takes a system of N particles and returns a system of N scalars, including N=1
    :return: diagnostic plot of vertical convergence of pi_t(f) to pi(f)
    """
    res = [np.mean(f(x_t)) for x_t in chain]
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.axhline(exact_low, linestyle='dashed')
    ax.axhline(exact_high, linestyle='dashed')
    fig.show()
    return fig, ax

# noinspection DuplicatedCode
def horizontal_convergence_plot(chain, f, exact_low, exact_high):
    """
    :param chain: chain[t] is the system of particles at time t in parallel MCMC
    :param f: diagnostic function that takes a system of N particles and returns a system of N scalars, including N=1
    :return: diagnostic plot of vertical convergence of 1/T (f(x_1) + ... + f(x_T) to pi(f)
    """
    index = np.random.randint(low=0, high=len(chain[0]))
    res = [f(pointset[[index]])[0] for pointset in chain]
    res = np.cumsum(res)/(np.arange(len(res)) + 1)
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.axhline(exact_low, linestyle='dashed')
    ax.axhline(exact_high, linestyle='dashed')
    fig.show()
    return fig, ax

def diagplot(x: np.ndarray, truepdf: Callable[[np.ndarray], np.ndarray] = None, inflate: float=1, no_points:int=500, display_limits: Tuple[float, float]=(None, None), mode='kde'):
    # Visual plots to see if things work.
    fig, ax = plt.subplots()
    if mode=='kde':
        sns.kdeplot(x, ax=ax)
        kde_line = ax.get_lines()[0]
    elif mode=='histogram':
        _ = ax.hist(x, density=True, bins='auto')
        kde_line=None
    else:
        raise ValueError('Unknown plot mode.')

    xlim = np.array(ax.get_xlim())
    xlim = xlim.mean() + (xlim - xlim.mean()) * inflate
    x = np.linspace(start=xlim[0], stop=xlim[1], num=no_points)
    y = truepdf(x) if truepdf is not None else np.array([0] * no_points)
    real_line, = ax.plot(x, y)

    if display_limits[0] is not None:
        ax.set_xlim(*display_limits)

    if mode=='kde':
        plt.legend([kde_line, real_line], ['kde', 'real'])
    fig.show()

