#tested
from particles import smc_samplers as ssp, resampling as rs
import numpy as np
from particles import distributions as dists
from typing import List, Callable, Any, Tuple
from collections import defaultdict
from libs_new.mcmc_new import MetropolisHastingsv2, MCMC
from libs_new import utils as ut

def log_factorial(d):
    return np.sum(np.log(np.arange(d) + 1))

class LatinSquaresProblem(ssp.StaticModel):
    def __init__(self, d):
        prior = UOPS(d=d)
        data = np.array([0])
        ssp.StaticModel.__init__(self, data, prior)
        self.d = d
        self.last_ep = (self.d ** 2) * np.log(self.d) - d * log_factorial(d) - np.log(1e-16)
        self.cpu = 0

    def logpyt(self, theta, t):
        self.cpu += len(theta)/(self.d*1000)
        return -np.array([sq.A for sq in theta]) * self.last_ep + self.prior.log_no_perm

class UOPS(dists.ProbDist):
    """
    Uniform Distribution on Permutation Squares
    """
    def __init__(self, d):
        self.d = d
        self.log_no_perm = d * log_factorial(d)

    def logpdf(self, x):
        return -self.log_no_perm

    def rvs(self, size=None):
        arr = np.array([self.rvs_one() for _ in range(size)])
        return arr

    def ppf(self, u):
        return NotImplementedError

    def rvs_one(self):
        #tested
        arr = np.array([np.random.permutation(self.d) for _ in range(self.d)])
        return PermutationSquare(PermutationSquare.to_columns(arr))

class PermutationSquare:
    """
    A permutation square is a square of size N x N whose rows are permutations of {0, 1, ..., N-1} (but not necessarily columns).
    """
    def __init__(self, columns: List):
        """
        A permutation square is represented by a list of columns. Each column is a list of integers taking values in {0, 1, ..., N-1}.
        """
        self.columns = columns
        self._A = None
        self.As = [None] * len(columns)
        self.d = len(columns)

    def __repr__(self):
        return 'PermutationSquare:' + self.__class__.to_numpy(self.columns).__repr__()

    def proposal(self):
        #tested
        r = np.random.randint(low=0, high=self.d)
        c1, c2 = np.random.choice(self.d, size=2, replace=False)
        new_columns = self.columns.copy()
        new_columns[c1], new_columns[c2] = self.columns[c1].copy(), self.columns[c2].copy()
        new_columns[c1][r], new_columns[c2][r] = new_columns[c2][r], new_columns[c1][r]
        new = self.__class__(columns=new_columns)
        new.As = self.As.copy()
        new.As[c1], new.As[c2] = None, None
        return new

    @property
    def A(self):
        #tested
        if self._A is None:
            for i, v in enumerate(self.As):
                if v is None:
                    # noinspection PyTypeChecker
                    self.As[i] = self.__class__.calculate_A(self.columns[i])
            self._A = sum(self.As)
        return self._A

    @staticmethod
    def to_columns(x: np.ndarray):
        #tested
        return x.T.tolist()

    @staticmethod
    def to_numpy(x: List):
        #tested
        return np.array(x).T

    @staticmethod
    def calculate_A(x: List[int]):
        #updated to new version and tested
        d = defaultdict(int)
        for item in x:
            d[item] += 1
        return sum([v**2 for v in d.values()]) - len(x)

    def __eq__(self, other):
        # tested
        return np.allclose(self.__class__.to_numpy(self.columns), self.__class__.to_numpy(other.columns))

    def __hash__(self):
        # tested
        sin = np.sin(np.arange(self.d) + 1).reshape((self.d, 1))
        cos = np.cos(np.arange(self.d) + 1).reshape((1, self.d))
        multiplier = sin @ cos
        res = np.abs(np.sum(self.__class__.to_numpy(self.columns) * multiplier))
        res = (res-int(res)) * 1000
        res = (res-int(res)) * 1e10
        return int(res)

class MetropolisOnLatinSquares(MetropolisHastingsv2):
    def proposal(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(x)), np.array([s.proposal() for s in x])

    @classmethod
    def initialize_from_ssp(cls, x: np.ndarray, w: rs.Weights, info: Callable[[np.ndarray, str], Any]) -> 'MCMC':
        # noinspection PyTypeChecker
        return cls(uld=ut.function_with_fixed_arguments(info, fixed_positional_arguments={1: 'uld'}))