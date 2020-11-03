import datetime
from collections import namedtuple
from particles import resampling as rs
from particles import utils
import numpy as np
from tqdm import tqdm
import typing
from inspect import signature
from abc import ABC, abstractmethod
from libs_new import utils as ut

class FeynmanKacNew(ABC):
    """
    A new kind of FeynmanKac model which generalizes particles.FeynmanKac
    """
    def __init__(self):
        # noinspection PyTypeChecker
        self.pf_debug_access: SMCNew = None # allow accessing `pf`. Should only be used for debugging or other exceptional circumstances (like adaptive FeynmanKac models that depend on the current approximating particles).

    @abstractmethod
    def M0(self, N: int) -> np.ndarray:
        ...

    @abstractmethod
    def M(self, t:int, xp: np.ndarray, w: rs.Weights) -> typing.Tuple[np.ndarray, rs.Weights]:
        """
        Take old particles and old weights.
        Return new particles and new weights.
        Weights are represented by Weight objects of the resampling module.
        For equal weights, do not use rs.Weights() but rather rs.Weights(np.zeros(N)).
        The normalization (or not) of returned weights are not important here.
        """
        ...

    @abstractmethod
    def logG(self, t: int, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def done(self, smc: 'SMCNew') -> bool:
        ...

    def afterlogG(self, pf: 'SMCNew'):
        pass

class SMCNew:
    # tested
    """
    SMC class that goes with FeynmanKacNew (i.e. equivalence of particles.SMC for particles.FeynmanKac).
    """
    def __init__(self, fk: FeynmanKacNew, N, max_memory_in_MB: float=np.inf, seed=None, verbose=False, print_datetime=True):
        self.fk = fk
        self.fk.pf_debug_access = self
        self.fk.verbose = verbose
        self.N = N
        self.seed = seed

        self.X = None
        self.t = 0
        self.logLt = 0
        self.loglt = []
        self.wgts = rs.Weights(np.zeros(N))

        self.verbose = verbose
        self.print_datetime = print_datetime
        self.dt_identifier = None
        if print_datetime:
            self.dt_identifier = str(datetime.datetime.now())
            print('Identifier {} with seed {} started'.format(self.dt_identifier, self.seed), flush=True)

        self.max_memory_in_MB = max_memory_in_MB
        ut.memory_tracker_add(0)

    def __next__(self):
        # X is X_{t-1}
        # wgts is W_{t-1}
        # X and wgts approximate Q_{t-1}
        if self.fk.done(self):
            if self.print_datetime:
                print('Identifier {} with seed {} terminated at {}'.format(self.dt_identifier, self.seed, str(datetime.datetime.now())), flush=True)
            raise StopIteration
        if self.t == 0:
            if self.seed:
                np.random.seed(self.seed)
            self.generate_particles()
        else:
            self.resample_move()
        # At this moment: X and wgts approximate Q_{t-1} M_t
        self.reweight_particles()
        self.fk.afterlogG(pf=self)
        ut.memory_tracker_add(self.t)
        if ut.memory_tracker[self.t] > self.max_memory_in_MB:
            raise MemoryError
        self.t += 1
        if self.verbose or self.print_datetime:
            print('id ' + str(self.seed) + ' iter:' + str(self.t), flush=True)

    def __iter__(self):
        return self

    @utils.timer
    def run(self):
        for _ in self:
            pass

    def generate_particles(self):
        self.X = self.fk.M0(self.N)

    def resample_move(self):
        self.X, self.wgts = self.fk.M(self.t, self.X, self.wgts)

    def reweight_particles(self):
        wgts = self.wgts
        logG = self.fk.logG(self.t, self.X)
        new_wgts = wgts.add(logG)
        loglt = rs.log_mean_exp(new_wgts.lw) - rs.log_mean_exp(wgts.lw)
        self.loglt += [loglt]
        self.logLt += loglt
        self.wgts = new_wgts

    def calculate_QT(self, phi: typing.Callable[[np.ndarray], np.ndarray]) -> float:
        return float(np.sum(self.wgts.W * phi(self.X)))

class CompactParticleHistory:
    # tested
    """
    Class to represent history of a standard particle filter, where time steps without resampling are collapsed into one step. Methods to estimate the variance of the normalizing constant and of the marginal distribution are also provided.
    """
    def __init__(self):
        self._logG: typing.List[np.ndarray] = list()
        self._A: typing.List[np.ndarray] = list() # ancestors for particles at time t are self._A[t]
        # noinspection PyTypeChecker
        self._X_last: np.ndarray = None

    def add(self, lw: np.ndarray, ancestor: typing.Union[np.ndarray, None], last_particles: np.ndarray):
        """
        Add a new step to the history. At t=0, ancestor should be set to `None`.
        """
        if self.T == 0 and ancestor is not None:
            raise ValueError
        self._logG.append(lw)
        if ancestor is not None:
            assert None not in ancestor
            assert np.min(ancestor) >= 0
            assert np.max(ancestor) <= self.N - 1
            assert len(ancestor) == self.N
        # noinspection PyTypeChecker
        self._A.append(ancestor)
        self._X_last = last_particles

    def modify_last(self, new_lw: np.ndarray, last_particles: np.ndarray, lw_update_mode: str):
        if lw_update_mode == 'add':
            self._logG[-1] = self._logG[-1] + new_lw
        elif lw_update_mode == 'replace':
            self._logG[-1] = new_lw
        else:
            raise ValueError('Unknown lw_update_mode')
        self._X_last = last_particles

    def _estimate_var_logLT(self) -> float:
        """
        Estimate the variance of logLT using the method of Anthony Lee & Nick Whiteley
        """
        GT_tilde = np.exp(self._logG[-1] - np.max(self._logG[-1]))
        return self.__class__.V_hat(arr=GT_tilde/np.mean(GT_tilde), Eve=self._Eve, T=len(self._logG) - 1)

    def _estimate_var_QT(self, phi: typing.Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Estimate the variance of Q_T(phi) using the method of Anthony Lee & Nick Whiteley
        :param phi: mathematically, a function from X (the space of particles) to R. Programatically, a function that takes N particles and return N floats.
        """
        GT_tilde = np.exp(self._logG[-1] - np.max(self._logG[-1]))
        phi_ = phi(self._X_last)
        QT_phi = np.mean(GT_tilde * phi_)/np.mean(GT_tilde)
        phi_bar = phi_ - QT_phi
        return self.__class__.V_hat(arr=phi_bar*GT_tilde/np.mean(GT_tilde), Eve=self._Eve, T=len(self._logG)-1)

    def extract(self, t1:int, t2:int) -> 'CompactParticleHistory':
        # tested
        """
        Extract the particle history between two times `t1` (included) and `t2` (excluded)
        """
        if t1 < 0 or t2 > self.T:
            raise ValueError
        res = self.__class__()
        res._logG = self._logG[t1:t2]
        res._A = self._A[t1:t2]
        # noinspection PyTypeChecker
        res._A[0] = None
        if t2 >= len(self._logG):
            res._X_last = self._X_last
        # noinspection PyTypeChecker
        return res

    def estimate_var_logLT(self, no_last_block:int=None) -> float:
        # tested
        """
        :param no_last_block: this argument should be understood as `block_size` but kept as `no_last_block` for compatibility.
        """
        res = 0
        iterator = _CompactParticleHistoryBackwardIterator(self, 2 if no_last_block==1.5 else no_last_block)
        for i, cph in enumerate(iterator):
            cph: CompactParticleHistory
            res += cph._estimate_var_logLT()
            if i == 0 and no_last_block == 1.5:
                iterator.block_size = 1
        return res

    def estimate_var_QT(self, phi: typing.Callable[[np.ndarray], np.ndarray], no_last_block:int=None) -> float:
        # tested
        if no_last_block == 1.5:
            no_last_block = 2
        iter_object = iter(_CompactParticleHistoryBackwardIterator(self, no_last_block))
        last_cph: CompactParticleHistory = next(iter_object)
        return last_cph._estimate_var_QT(phi)

    def ESS_ratio(self, t:int) -> typing.Union[float, None]:
        if t >= len(self._logG):
            return None
        return ut.ESS_ratio(rs.Weights(lw=self._logG[t]))

    @property
    def _Eve(self) -> np.ndarray:
        # tested
        a = np.arange(self.N)
        for A in reversed(self._A):
            if A is not None:
                a = A[a]
        return a

    @property
    def N(self) -> int:
        return len(self._logG[-1])

    @staticmethod
    def V_hat(arr:np.ndarray, Eve: np.ndarray, T:int) -> float:
        # tested
        """
        Calculate the formula (4) in the article of Anthony Lee and Nick Whiteley.
        """
        sums_of_arr_grouped_by_Eve = np.zeros(len(Eve))
        for element, eve in zip(arr, Eve):
            sums_of_arr_grouped_by_Eve[eve] += element
        complicated_sum_of_squares = np.sum(sums_of_arr_grouped_by_Eve**2) # calculate the complicated sum of squares in the formula (4)
        mean_squared = np.mean(arr) ** 2
        N = len(arr)
        p = (N / (N - 1)) ** (T + 1)
        return mean_squared * (1 - p) + p / (N ** 2) * complicated_sum_of_squares

    @property
    def logLT(self) -> float:
        """
        Calculate the log of the normalizing constant of the model. Can be useful for debugging (i.e., by comparing its with the estimates returned by the SMCNew executor).
        """
        return sum([rs.log_mean_exp(s) for s in self._logG])

    def calculate_QT(self, phi: typing.Callable[[np.ndarray], np.ndarray]) -> float:
        """Alternative way to calculate QT(phi). Used for debugging purpose
        """
        # noinspection PyTypeChecker
        return np.sum(rs.Weights(self._logG[-1]).W * phi(self._X_last))

    @property
    def T(self) -> int:
        return len(self._logG)

class _CompactParticleHistoryBackwardIterator:
    # tested
    def __init__(self, cph: CompactParticleHistory, block_size:int):
        self.cph = cph
        self.block_size = cph.T if (block_size is None) or (block_size == 0) else block_size

    def __iter__(self):
        self.t2 = self.cph.T
        self.t1 = None
        return self

    def __next__(self) -> CompactParticleHistory:
        if self.t2 == 0:
            raise StopIteration
        self.t1 = max(0, self.t2 - self.block_size)
        ret = self.cph.extract(self.t1, self.t2)
        self.t2 = self.t1
        return ret

class ClassicalFeynmanKac(ABC):
    """
    Represents a classical Feynman-Kac model
    """
    @abstractmethod
    def M0(self, N:int) -> np.ndarray:
        ...

    @abstractmethod
    def M(self, t:int, xp:np.ndarray) -> np.ndarray:
        """
        Represents a Markov kernel
        """
        ...

    @abstractmethod
    def logG(self, t:int, x: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def T(self) -> int:
        ...

class GenericParticleFilterPunctualLogging:
    def __init__(self):
        self.ESS_ratio, self.ESS, self.resampled = None, None, None

class GenericParticleFilterLogging:
    def __init__(self):
        self.punctual_logging: typing.Dict[int, GenericParticleFilterPunctualLogging] = dict()
        self.compact_particle_history = CompactParticleHistory()

class GenericParticleFilter(FeynmanKacNew):
    # tested
    """
    Wrapper to turn a mathematical `ClassicalFeynmanKac` object into an executable `FeynmanKacNew` object. Compact particle history is also calculated, from which variance estimates can be derived.
    """
    def __init__(self, fk_model: ClassicalFeynmanKac, ESSrmin:float, resampling_mode: str='multinomial'):
        super().__init__()
        self.fk_model = fk_model
        self.ESSrmin = ESSrmin
        self.resampling_mode = resampling_mode
        self.logging = GenericParticleFilterLogging()

    def M0(self, N: int) -> np.ndarray:
        res = self.fk_model.M0(N)
        self.logging.compact_particle_history.add(lw=np.zeros(N), ancestor=None, last_particles=res)
        return res

    def M(self, t: int, xp: np.ndarray, w: rs.Weights) -> typing.Tuple[np.ndarray, rs.Weights]:
        resampling_needed = ut.ESS_ratio(w) < self.ESSrmin
        self.logging.punctual_logging[t] = GenericParticleFilterPunctualLogging()
        self.logging.punctual_logging[t].ESS_ratio = ut.ESS_ratio(w)
        self.logging.punctual_logging[t].ESS = w.ESS
        self.logging.punctual_logging[t].resampled = resampling_needed
        if resampling_needed:
            ancestors = rs.resampling(self.resampling_mode, w.W)
            xp = xp[ancestors]
            w = rs.Weights(lw=np.zeros(len(xp)))
            self.logging.compact_particle_history.add(lw=w.lw, ancestor=ancestors, last_particles=xp)
        x = self.fk_model.M(t, xp)
        return x, w

    def logG(self, t: int, x: np.ndarray) -> np.ndarray:
        res = self.fk_model.logG(t, x)
        self.logging.compact_particle_history.modify_last(new_lw=res, last_particles=x, lw_update_mode='add')
        return res

    def done(self, smc: 'SMCNew') -> bool:
        res = smc.t > self.fk_model.T
        if res:
            assert np.allclose(smc.logLt, self.logging.compact_particle_history.logLT)
            self.logging.punctual_logging[smc.t] = GenericParticleFilterPunctualLogging()
            self.logging.punctual_logging[smc.t].ESS_ratio = ut.ESS_ratio(smc.wgts)
            self.logging.punctual_logging[smc.t].ESS = smc.wgts.ESS
        return res

    def __repr__(self):
        res = ''
        i = 1
        while self.logging.punctual_logging.get(i) is not None:
            if i > 1:
                res = res[:-1] + '; '
            res += 'M{i}: ESS_ratio={ess_ratio:.2f}, resampled={resample}.'.format(i=i, ess_ratio=self.logging.punctual_logging[i].ESS_ratio, resample=self.logging.punctual_logging[i].resampled)
            i += 1
        return res

class WeightCumulator:
    # should be assert-tested in classes where it is used, for instance WastelessSMC
    """A weight cumulator is an object that is updated on-the-fly as a `FeynmanKacNew` model is executed. It provides information on the current particle weights, but also, with its `normalize` method, information on the normalizing constant. It can also be used as a pseudo-`CompactParticleHistory`, because the `normalize` method breaks the calculation of the normalizing constant into multiple components.
    """
    def __init__(self):
        self.pre_normalize_weights: typing.List[np.ndarray]= [] # remember: weights are always expressed in log scale.
        self.normalize_information: typing.List[typing.Any] = []
        self.current_weights: np.ndarray = np.array([])
        self.current_particles: np.ndarray = np.array([])
        self._initialized = False

    def update_M(self, new_weights: rs.Weights, normalize:bool, normalize_info: typing.Any=None):
        if normalize:
            self.pre_normalize_weights.append(self.current_weights)
            self.normalize_information.append(normalize_info)
            self.current_weights = np.log(new_weights.W)
        else:
            normalized_new_weights = ut.log(new_weights.W)
            normalization_constant_of_current_weights = rs.log_sum_exp(self.current_weights)
            self.current_weights = normalized_new_weights + normalization_constant_of_current_weights

    def update_G(self, logGt: np.ndarray, x: np.ndarray):
        if len(self.current_weights) != len(logGt): # should only happen at initialization
            assert not self._initialized
            self.current_weights = np.zeros(len(logGt)) - np.log(len(logGt))
            self._initialized = True
        self.current_weights = self.current_weights + logGt
        self.current_particles = x

    @property
    def logLT(self) -> float:
        return sum([rs.log_sum_exp(arr) for arr in self.pre_normalize_weights + [self.current_weights]])

# def multiSMCNew(nruns=10, nprocs=0, out_func=None, **args):
#     """
#     Equivalence of particles.mutliSMC, for use with SMCNew.
#     """
#     def f(**kargs):
#         pf = SMCNew(**kargs)
#         pf.run()
#         return out_func(pf)
#
#     if out_func is None:
#         out_func = lambda x: x
#     return utils.multiplexer(f=f, nruns=nruns, nprocs=nprocs, seeding=True,
#                              **args)

def identity(x):
    return x

class _MultiSMCWrapFunction:
    def __init__(self, out_func: typing.Callable, kwargs: dict):
        self.out_func = out_func
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        pf = SMCNew(**self.kwargs)
        pf.run()
        return self.out_func(pf)

def multiSMCNew(nruns: int, nprocs: int, start_method:str='fork', out_func: typing.Callable=identity, **kwargs):
    """
    Equivalent of particles.multiSMC, for use with SMCNew
    """
    f = _MultiSMCWrapFunction(out_func, kwargs)
    return ut.parallelly_evaluate(f, parargs=[0] * nruns, n_cores=nprocs, start_method=start_method)

def multiSMCNewSingle(cls, params, N, nruns, verbose):
    # todo: rewrite!
    res = []
    for _ in tqdm(range(nruns), disable=not verbose):
        fk = cls(**params)
        pf = SMCNew(fk=fk, N=N)
        pf.run()
        res.append({'output':pf})
    return res

class SMCTwice:
    # tested
    """
    A class for running a particle filter twice: one first, pilot run for choosing certain parameters (par ex. no. of particles or whether to resample at certain steps), then a second, final run, based on parameters chosen on the pilot run.
    """
    def __init__(self, pilot_run_fk: FeynmanKacNew, N_pilot:int, final_run_creator: typing.Callable[[SMCNew], typing.Tuple[FeynmanKacNew, int]], seed=None, seed_final=None, verbose=False):
        """
        :param final_run_creator: a function that returns the final run FeynmanKac and the final run number of particles `N` based on the pilot run filter.
        """
        self.pilot_run_fk, self.final_run_creator, self.seed, self.seed_final, self.N_pilot, self.verbose = pilot_run_fk, final_run_creator, seed, seed_final, N_pilot, verbose
        self.pf_pilot, self.pf_final = None, None
        self.already_run = False

    def run(self):
        assert not self.already_run
        if self.seed:
            np.random.seed(self.seed)
        self.pf_pilot = SMCNew(fk=self.pilot_run_fk, N=self.N_pilot, verbose=self.verbose)
        self.pf_pilot.run()
        fk, N = self.final_run_creator(self.pf_pilot)
        self.pf_final = SMCNew(fk=fk, N=N, seed=self.seed_final, verbose=self.verbose)
        self.pf_final.run()
        self.already_run = True

    def __getattr__(self, item):
        if item in ['fk', 'N', 'X', 'logLt', 'loglt', 't', 'wgts']:
            assert self.already_run
            return getattr(self.pf_final, item)
        else:
            raise AttributeError('SMCTwice has no attribute ' + str(item))

SMCTwice_signature = [s for s in signature(SMCTwice).parameters]
for s in ['seed', 'seed_final', 'verbose']:
    SMCTwice_signature.remove(s)
SMCTwice_signature = namedtuple('SMCTwice_signature', SMCTwice_signature)

def multiSMCTwice(nruns=10, nprocs=0, out_func=None, **args):
    """
    Equivalence of particles.multiSMC, for use with SMCTwice.
    """
    def f(**kargs):
        pf = SMCTwice(**kargs)
        pf.run()
        return out_func(pf)

    if out_func is None:
        out_func = lambda x: x
    return utils.multiplexer(f=f, nruns=nruns, nprocs=nprocs, seeding=True,
                             **args)