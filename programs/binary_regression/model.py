from particles import smc_samplers as ssp
import numpy as np
from particles.distributions import ProbDist
from scipy.stats import cauchy
from scipy.optimize import linprog
from libs_new.utils import multiply_each_row_with

class BinaryRegression(ssp.StaticModel):
    #tested, much faster than previous version based on logpyt
    def __init__(self, data: str, prior: str):
        data = load_data(data)
        self.d = data.shape[1] - 1
        self.cpu = 0
        if prior == 'gaussian':
            scale = [20] + [5] * (self.d - 1)
            prior = GaussianPrior(scale=scale)
        elif prior == 'cauchy':
            scale = np.array([10] + [2.5] * (self.d - 1))
            prior = CauchyPrior(scale=scale)
        else:
            raise ValueError('Unknown prior distribution.')
        self.predictors_T = data[:, 1:].T
        self.response = data[:, 0]
        ssp.StaticModel.__init__(self, data=np.array([0]), prior=prior)

    def logpyt(self, theta, t):
        self.cpu += len(theta)/50000
        gram_matrix = theta @ self.predictors_T
        return np.sum(log_logit(gram_matrix * self.response), axis=1)

def load_data(name: str):
    return np.load(file='./programs/binary_regression/data/' + name + '.npy')

class CauchyPrior(ProbDist):
    # tested
    def __init__(self, scale):
        self.scale = np.array(scale)
        self.frozen_dists = [cauchy(loc=0, scale=s) for s in scale]
        self.cst = -np.log(np.pi * self.scale)

    def logpdf(self, x):
        # tested, faster than using frozen_dists.
        x_logpdf_fed = -np.log(1 + (x/self.scale)**2) + self.cst
        return np.sum(x_logpdf_fed, axis=1)

    def rvs(self, size=None):
        return np.array([self.frozen_dists[i].rvs(size=size) for i in range(len(self.scale))]).T

    def ppf(self, u):
        raise NotImplementedError

    @property
    def dim(self):
        return len(self.scale)

class GaussianPrior(ProbDist):
    def __init__(self, scale):
        self.scale = np.array(scale)
        self.var = self.scale**2
        self.cst = -1/2 * np.sum(np.log(2 * np.pi * self.var))

    def logpdf(self, x):
        #tested
        return self.cst - np.sum(x**2/(2*self.var), axis=1)

    def ppf(self, u):
        raise NotImplementedError

    @property
    def dim(self):
        return len(self.scale)

    def rvs(self, size=None):
        #tested
        if size is None:
            size = 1
        return np.random.normal(size=(size, len(self.scale))) * self.scale

def log_logit(arr):
    # tested for correctness
    # a little bit faster than old versions.
    """
    :return: -log(1+exp(-arr)), calculated safely
    """
    delta = np.abs(arr)
    return 1/2 * (arr - delta) - np.log1p(np.exp(-delta))

def one_sided(x: np.ndarray, maxiter=None):
    """
    For an array x of shape nxp, returns whether all row vectors of `x` lie on some same half-space of R^p which contains 0.
    """
    # tested
    n, p = x.shape
    # noinspection PyTypeChecker
    res1 = linprog(c = np.array([0] * p), A_ub=x, b_ub=np.array([0] * n), A_eq=np.array([[1] + [0] * (p-1)]), b_eq=np.array([[1]]), bounds=(None, None), options=None if maxiter is None else dict(maxiter=maxiter))
    # noinspection PyTypeChecker
    res2 = linprog(c = np.array([0] * p), A_ub=x, b_ub=np.array([0] * n), A_eq=np.array([[1] + [0] * (p-1)]), b_eq=np.array([[-1]]), bounds=(None, None), options=None if maxiter is None else dict(maxiter=maxiter))
    if res1.status == 2 and res2.status == 2:
        return False
    if (res1.status in (1,4)) or (res2.status in (1,4)):
        raise RuntimeError('Numerical difficulties encountered.')
    return True

def separated(data_name:str, maxiter=None):
    """
    Test whether a dataset is completely separated or not.
    """
    data = load_data(data_name)
    y = data[:,0]
    x = data[:,1:]
    x = multiply_each_row_with(x, y)
    return one_sided(x, maxiter=maxiter)
