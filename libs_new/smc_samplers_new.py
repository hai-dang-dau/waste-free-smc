from libs_new.cores_new import FeynmanKacNew
from particles import resampling as rs
import numpy as np
from libs_new.cores_new import SMCNew, WeightCumulator, CompactParticleHistory
import gc
from tqdm import tqdm
from abc import ABC, abstractmethod
import typing as tp
from libs_new import utils as ut
from libs_new import mcmc_new
from libs_new.variance_estimator import MCMC_variance, TargetedReproducer

ut.shut_off_numba_warnings()

class SMCSamplerModel(ABC):
    # tested
    def __init__(self, mcmc_engine: tp.Type[mcmc_new.MCMC]):
        self.mcmc_engine = mcmc_engine
        self.mcmc_object: tp.Dict[int, mcmc_new.MCMC] = dict()
        # noinspection PyTypeChecker
        self.pf_debug_access: SMCNew = None # allow accessing `pf` to get information about current particles and current weights. Should only be used under special circumstances (like adaptive algorithms).

    @abstractmethod
    def M0(self, N: int) -> np.ndarray:
        ...

    @abstractmethod
    def logG(self, t: int, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def mcmc_info(self, t:int, x: np.ndarray, req: str):
        """
        Give information that will be required by the MCMC kernel. Typically, if `req` is `uld`, the function returns the unnormalized log density of `Q_{t-1}`; if `req` is `gld`, the function returns the gradient of the log density. Some of this may not be necessarily implemented, depending on the MCMC kernel.
        """
        ...

    def MCMC(self, t:int, x: np.ndarray) -> np.ndarray:
        """
        :param x: a system of `N` particles (represented by np.array or similar)
        :return: a new system of `N` particles where an MCMC step was applied on each of the input `N` particles. The invariant measure is `Q_{t-1}`.
        """
        return self.mcmc_object[t].step(x)

    def MCMC_initializer(self, t:int, x: np.ndarray, w: rs.Weights) -> None:
        """
        Initialize the MCMC kernel at time `t` using an approximate sample of `Q_{t-1}` provided by the sampler.
        """
        #self.mcmc_object[t] = self.mcmc_engine.initialize_from_ssp(x=x, w=w, info=lambda _x, _req: self.mcmc_info(t, _x, _req)) # not pickable!
        # noinspection PyTypeChecker
        self.mcmc_object[t] = self.mcmc_engine.initialize_from_ssp(x=x, w=w,
                                                                   info=ut.function_with_fixed_arguments(self.mcmc_info, {0:t}))

    @abstractmethod
    def extend(self, t: int, xp: np.ndarray) -> np.ndarray:
        """
        To be used in case where the Markov step `M_t` includes not only MCMC kernels that keep invariant Q_{t-1}, but also other things (for example, adding one dimension to the problem).
        """
        ...

    @abstractmethod
    def done(self, smc: SMCNew) -> bool:
        ...

    def diag_function_for_adaptive_MCMC(self, t:int, x:np.ndarray) -> np.ndarray:
        """A function that turns particles into scalars, used for choosing MCMC length automatically if requested. Defaults to logG(t-1), should be overriden by a less expensive function if available."""
        return self.logG(t-1, x)

    def MCMC_free_memory(self, t:int) -> None:
        self.mcmc_object[t].free_memory()

class FKSMCSamplerNew(FeynmanKacNew, ABC):
    # tested
    def __init__(self, model: SMCSamplerModel):
        super().__init__()
        self.model = model
        self.verbose = False
        # Mutable attributes
        self.current_post_rejuvenate_w: rs.Weights = rs.Weights()

    def M0(self, N):
        self.model.pf_debug_access = self.pf_debug_access
        res = self.model.M0(N)
        return res

    def logG(self, t, x):
        return self.model.logG(t, x)

    def done(self, smc):
        return self.model.done(smc)

    def M(self, t, xp, w):
        if self.need_rejuvenate(t, xp, w):
            self.model.MCMC_initializer(t, xp, w)
            xp, w = self.rejuvenate(t, xp, w)
            self.model.MCMC_free_memory(t)
        self.current_post_rejuvenate_w = w
        xp = self.model.extend(t, xp)
        return xp, w

    @abstractmethod
    def need_rejuvenate(self, t, xp, w: rs.Weights) -> bool:
        ...

    @abstractmethod
    def rejuvenate(self, t: int, xp: np.ndarray, w: rs.Weights) -> tp.Tuple[np.ndarray, rs.Weights]:
        ...

class ClassicalSMCSamplerLogging:
    """
    Log from a M point of view
    """
    # tested
    def __init__(self):
        self.inner_ESS_ratio: tp.List[float] = []
        self.resampling_mode: tp.List[bool] = []
        self.compact_particle_history = CompactParticleHistory()

class ClassicalSMCSampler(FKSMCSamplerNew):
    # tested
    def __init__(self, model: SMCSamplerModel, k:int, ESSrmin: float, resampling_scheme: str='systematic'):
        super().__init__(model=model)
        self.k, self.ESSrmin = k, ESSrmin
        self.logging = ClassicalSMCSamplerLogging()
        self.resampling_scheme = resampling_scheme

    def need_rejuvenate(self, t:int, xp, w: rs.Weights) -> bool:
        self.logging.inner_ESS_ratio.append(ut.ESS_ratio(w))
        res = w.ESS < len(xp) * self.ESSrmin
        self.logging.resampling_mode.append(res)
        return res

    def rejuvenate(self, t, xp, w):
        ancestors = rs.resampling(self.resampling_scheme, w.W)
        x = xp[ancestors]
        for _ in tqdm(range(self.k), disable= not self.verbose):
            x = self.model.MCMC(t, x)
        self.logging.compact_particle_history.add(lw=np.zeros(len(x)), ancestor=ancestors, last_particles=x)
        ut.memory_tracker_add(t)
        return x, rs.Weights(lw=np.zeros(len(xp)))

    def done(self, smc):
        res = super().done(smc)
        if res:
            assert np.allclose(self.pf_debug_access.logLt, self.logging.compact_particle_history.logLT)
            # noinspection PyTypeChecker
            self.logging.resampling_mode.append(None)
            self.logging.inner_ESS_ratio.append(ut.ESS_ratio(smc.wgts))
        return res

    def __repr__(self):
        res = ''
        for i, (ess_ratio, rejuvenate) in enumerate(ut.zip_with_assert(self.logging.inner_ESS_ratio, self.logging.resampling_mode)):
            res += 'M{ip1}: ess_ratio={ess_ratio:.2f}, rejuvenate={rejuvenate}\n'.format(ip1=i+1, ess_ratio=ess_ratio, rejuvenate=rejuvenate)
        return res[:-1]

    def logG(self, t, x):
        res = super().logG(t, x)
        if t == 0:
            self.logging.compact_particle_history.add(lw=res, ancestor=None, last_particles=x)
        else:
            self.logging.compact_particle_history.modify_last(new_lw=res, last_particles=x, lw_update_mode='add')
        return res

    def compute_variance_QT_via_Lee_and_Whiteley(self, phi: tp.Callable[[np.ndarray], np.ndarray],
                                                 no_last_block: int = None) -> float:
        assert np.allclose(self.logging.compact_particle_history.calculate_QT(phi), self.pf_debug_access.calculate_QT(phi))
        return self.logging.compact_particle_history.estimate_var_QT(phi, no_last_block)

    def compute_variance_logLT_via_Lee_and_Whiteley(self, no_last_block:int=None) -> float:
        return self.logging.compact_particle_history.estimate_var_logLT(no_last_block)

class MultipleViewArray:
    # tested
    def __init__(self, flattened_view: np.ndarray, M:int, M1:int):
        """
        :param M1: size of each particle island
        """
        self.flattened_view = flattened_view
        self.M1 = M1
        self.M2 = M // M1
        self.P = len(flattened_view) // M
        assert M % M1 == 0
        assert len(flattened_view) % M == 0

    @ut.cached_property
    def detailled_view(self) -> np.ndarray:
        """The view indexing is (p,m2,m1) where M1 is the size of each island and M2 is the number of islands.
        """
        return np.reshape(self.flattened_view, (self.P, self.M2, self.M1, *self.flattened_view.shape[1:]))

class MultipleViewWeights:
    # tested
    def __init__(self, flattened_lw: np.ndarray, M:int, M1:int):
        self.flattened_lw = flattened_lw
        # Base-tilde decomposition: x = exp(b)*t
        self.base = np.max(flattened_lw)
        self.flattened_tilde = np.exp(flattened_lw - self.base)
        self.detailled_view_tilde = MultipleViewArray(flattened_view=self.flattened_tilde, M=M, M1=M1).detailled_view

    @ut.cached_property
    def outer_lw(self) -> np.ndarray:
        return ut.log(np.mean(self.detailled_view_tilde, axis=(0,2))) + self.base

    @ut.cached_property
    def outer_weights(self) -> rs.Weights:
        return rs.Weights(lw=self.outer_lw)

    @ut.cached_property
    def get_samplable_weights_inside_island(self) -> np.ndarray:
        """Should be used like a function: `get_samplable_weights_inside_island[m2]`.
        """
        island_last = np.transpose(self.detailled_view_tilde, (0,2,1))
        with np.errstate(invalid='ignore'):
            res_last = island_last/np.sum(island_last, axis=(0,1))
        return np.transpose(res_last, (2,0,1))

class AdaptivelySampleStartingPoints:
    # tested
    """
    Helper function to rejuvenate particles in Wasteless SMC. The purpose is to get, from particles `xp` with weights `w` at time t-1, `M` starting points (and possibly weights) for the `M` MCMC chains of the Markov kernel at time `t`.
    """
    def __init__(self, xp: np.ndarray, w: rs.Weights, M:int, M1:int, ESSrmin_outer:float, inner_resampling_mode: str, outer_resampling_mode:str):
        self.x = MultipleViewArray(flattened_view=xp, M=M, M1=M1)
        self.w = MultipleViewWeights(flattened_lw=w.lw, M=M, M1=M1)
        self.ESSrmin_outer = ESSrmin_outer
        self.M1 = M1
        self.M2 = M // M1
        self.inner_resampling_mode, self.outer_resampling_mode = inner_resampling_mode, outer_resampling_mode

    @ut.cached_property
    def final_starting_points(self) -> np.ndarray:
        # tested
        p_list = []
        m2_list = []
        m1_list = []
        for m2 in self.chosen_outer_ancestors:
            p_, m1_ = ut.array_resampling(resampling_scheme=self.inner_resampling_mode, W=self.w.get_samplable_weights_inside_island[m2], M=self.M1)
            p_list.append(p_)
            m1_list.append(m1_)
            m2_list = m2_list + [m2] * self.M1
        return self.x.detailled_view[np.r_[tuple(p_list)], np.r_[tuple(m2_list)], np.r_[tuple(m1_list)]]

    @ut.cached_property
    def chosen_outer_ancestors(self) -> np.ndarray:
        if self.outer_resampling_needed:
            return rs.resampling(scheme=self.outer_resampling_mode, W=self.w.outer_weights.W)
        else:
            return np.arange(self.M2)

    @ut.cached_property
    def outer_resampling_needed(self) -> bool:
        outer_ess_ratio = ut.ESS_ratio(self.w.outer_weights)
        return outer_ess_ratio < self.ESSrmin_outer

    def final_inner_lw(self, P:int) -> np.ndarray:
        if self.outer_resampling_needed:
            outer_lw = np.zeros(self.M2)
        else:
            outer_lw = self.w.outer_lw
        # noinspection PyTypeChecker
        return np.array(np.repeat(outer_lw, self.M1).tolist() * P)

class WastelessSMCLogging:
    # assert-tested
    resampling_mode_code = {0: 'no_resampling', 1:'inner_only', 2:'inner_and_outer'}

    def __init__(self):
        self.resampling_mode: tp.List[int] = []
        self.inner_ESS_ratio: tp.List[float] = []
        self.outer_ESS_ratio: tp.List[float] = []
        self.chain_length: tp.List[int] = []
        self.thinning: tp.List[int] = []
        self.weight_cumulator = WeightCumulator()
        # noinspection PyTypeChecker
        self.M: int=None
        # noinspection PyTypeChecker
        self.M1: int=None
        # noinspection PyTypeChecker
        self.M2: int=None
        self.done = False

    @ut.cached_property
    def compact_particle_history(self) -> CompactParticleHistory:
        assert self.done
        res = CompactParticleHistory()
        for ancestors in self.weight_cumulator.normalize_information:
            assert isinstance(ancestors, np.ndarray)
        for normalized_logG, ancestors in ut.zip_with_assert(self.weight_cumulator.pre_normalize_weights + [self.weight_cumulator.current_weights], [None] + self.weight_cumulator.normalize_information):
            res.add(lw=MultipleViewWeights(flattened_lw=normalized_logG, M=self.M, M1=self.M1).outer_lw + np.log(len(normalized_logG)), ancestor=ancestors, last_particles=np.array([None]* self.M2))
        last_particles = MultipleViewArray(self.weight_cumulator.current_particles, M=self.M, M1=self.M1).detailled_view
        last_particles = np.transpose(last_particles, axes=(1,0,2, *np.arange(len(last_particles.shape))[3:]))
        res.modify_last(new_lw=np.zeros(self.M2), last_particles=last_particles, lw_update_mode='add')
        assert np.allclose(res.logLT, self.weight_cumulator.logLT)
        return res

class HistoryBetweenTwoOuterResampling:
    """
    Helper to estimate outer ESS ratio in a conservative but low-varianced function. Recall that outer ESS ratio reduces to the variance of the estimate of the log-normalizing constant of a particle filter.
    """
    # tested
    def __init__(self, M1:int, M:int):
        self.M1 = M1
        self.M = M
        # Mutable attributes
        self.collapsed_logGs: tp.List[np.ndarray] = list() # time steps without resampling are collapsed into one step
        self.var_loglts: tp.List[float] = list()
        self.current_collapsed_logG: np.ndarray = np.array([])

    @staticmethod
    def var_loglt(logG: np.ndarray, M1:int, M:int):
        # tested
        P = len(logG) // M
        G = np.exp(logG - np.max(logG)).reshape((P, M))
        varG = np.var(G)
        inflation = np.var(np.sum(G,axis=0)/np.sqrt(P))/varG
        ess = M1 * max(P/inflation - 1, P/(2*inflation))
        return varG/ess/(np.mean(G)**2)

    def add_new(self, logG: np.ndarray):
        if len(self.current_collapsed_logG) > 0:
            self.collapsed_logGs.append(self.current_collapsed_logG)
            self.var_loglts.append(self.__class__.var_loglt(self.current_collapsed_logG, self.M1, self.M))
        self.current_collapsed_logG = logG

    def update_last(self, logG:np.ndarray):
        self.current_collapsed_logG = self.current_collapsed_logG + logG

    @property
    def var_log_LT(self) -> float:
        return sum(self.var_loglts) + self.__class__.var_loglt(self.current_collapsed_logG, self.M1, self.M)

    @property
    def logLTs_of_M1(self) -> tp.Sequence:
        """
        returns estimators of (local) `logLT` by M2 particle filters, each containing M1 particles (mainly for debugging purpose)
        """
        logG_of_islands = [MultipleViewWeights(flattened_lw=arr, M=self.M, M1=self.M1).outer_lw for arr in self.collapsed_logGs + [self.current_collapsed_logG]]
        return sum(logG_of_islands)

    @property
    def logLT(self) -> float:
        """
        returns the local logLT (for debugging)
        """
        # noinspection PyTypeChecker
        return rs.log_mean_exp(self.logLTs_of_M1)

    @property
    def outer_ess_ratio(self) -> float:
        return ut.get_ESS_ratio(self.var_log_LT)

class WastelessSMC(FKSMCSamplerNew):
    # tested beyond doubts.
    def __init__(self, model: SMCSamplerModel, k:tp.Union[int, None], M:int, M1:int, ESSrmin_inner: float=0.5, ESSrmin_outer: float=0.5, inner_resampling_scheme: str='systematic', outer_resampling_scheme: str='multinomial', outer_essr_calculation_mode='standard'):
        """
        :param outer_essr_calculation_mode: one may use `conservative` instead of `standard` for a conservative but lower-varianced estimation of outer ESS.
        """
        super().__init__(model=model)
        self.k, self.M_resample, self.ESSrmin_inner, self.M1 = k, M, ESSrmin_inner, M1
        self.ESSrmin_outer = ESSrmin_outer
        assert M % M1 == 0
        self.M2 = self.M_resample // self.M1
        self.logging = WastelessSMCLogging()
        self.logging.M = self.M_resample; self.logging.M1 = self.M1; self.logging.M2 = self.M2
        self.inner_resampling_scheme, self.outer_resampling_scheme = inner_resampling_scheme, outer_resampling_scheme
        self.outer_ess_r_calc_mode = outer_essr_calculation_mode
        # === Mutable attributes (except logging): do not forget to update these
        self.inner_lw_just_after_last_rejuvenate: np.ndarray = ...
        self.histories_between_two_outer_resampling: tp.List[HistoryBetweenTwoOuterResampling] = list()
        self.last_resampling_mode = 2

    def M0(self, N):
        res = super().M0(N)
        self.inner_lw_just_after_last_rejuvenate = np.zeros(N)
        return res

    def _inner_ess_ratio(self, current_w: rs.Weights) -> float:
        """Calculate the ESS ratio between the most recently rejuvenated distribution and the current distribution, using the difference between the weights just after rejuvenation and the current weights.
        """
        assert len(self.inner_lw_just_after_last_rejuvenate) == len(current_w.lw)
        return ut.get_ESS_ratio(ut.chi_squared_distance(self.inner_lw_just_after_last_rejuvenate, current_w.lw))

    def need_rejuvenate(self, t, xp, w: rs.Weights) -> bool:
        inner_ess_ratio = self._inner_ess_ratio(w)
        self.logging.inner_ESS_ratio.append(inner_ess_ratio)
        self.logging.outer_ESS_ratio.append(ut.ESS_ratio(MultipleViewWeights(flattened_lw=w.lw, M=self.M_resample, M1=self.M1).outer_weights))
        res = inner_ess_ratio < self.ESSrmin_inner
        if not res:
            self.logging.resampling_mode.append(0)
            self.logging.weight_cumulator.update_M(new_weights=w, normalize=False)
            self.logging.chain_length.append(0)
            self.logging.thinning.append(0)
            self.last_resampling_mode = 0
        return res

    def rejuvenate(self, t: int, xp: np.ndarray, w: rs.Weights) -> tp.Tuple[np.ndarray, rs.Weights]:
        if self.outer_ess_r_calc_mode == 'standard':
            essrmin_outer_command = self.ESSrmin_outer # we let the resampler determine whether to outer-resample or not
        elif self.outer_ess_r_calc_mode == 'conservative':
            essrmin_outer_command = int(self.histories_between_two_outer_resampling[-1].outer_ess_ratio < self.ESSrmin_outer) # we decide in avance
        else:
            raise ValueError('Unknown method to calculate outer ESS ratio.')
        resampler = AdaptivelySampleStartingPoints(xp=xp, w=w, M=self.M_resample, M1=self.M1, ESSrmin_outer=essrmin_outer_command, inner_resampling_mode=self.inner_resampling_scheme, outer_resampling_mode=self.outer_resampling_scheme)
        starting_points = resampler.final_starting_points
        self.logging.resampling_mode.append(1 if not resampler.outer_resampling_needed else 2)
        self.last_resampling_mode = 1 if not resampler.outer_resampling_needed else 2
        x = self.apply_MCMC_then_flatten(t=t, starting_points=starting_points, target_N = len(xp), starting_lw=resampler.final_inner_lw(1))
        P = len(x) // self.M_resample
        new_inner_weights = rs.Weights(resampler.final_inner_lw(P))
        self.inner_lw_just_after_last_rejuvenate = new_inner_weights.lw
        self.logging.weight_cumulator.update_M(new_weights=new_inner_weights, normalize=resampler.outer_resampling_needed, normalize_info=resampler.chosen_outer_ancestors)
        return x, new_inner_weights

    # noinspection PyUnusedLocal
    def apply_MCMC_then_flatten(self, t:int, starting_points: np.ndarray, target_N: int, starting_lw:np.ndarray) -> np.ndarray:
        # tested
        assert len(starting_points) == self.M_resample
        P = target_N // self.M_resample
        res = [starting_points]
        x = starting_points
        for _ in tqdm(range(P-1), disable=not self.verbose):
            for __ in range(self.k):
                x = self.model.MCMC(t, x)
            res.append(x)
        ut.memory_tracker_add(t)
        assert sum([len(s) for s in res]) == target_N
        self.logging.thinning.append(self.k)
        self.logging.chain_length.append((P-1) * self.k + 1)
        return self.collector(res)

    @staticmethod
    def collector(l):
        gc.collect()
        return np.r_[tuple(l)]

    def logG(self, t, x):
        res = super().logG(t, x)
        self.logging.weight_cumulator.update_G(res, x)
        if self.last_resampling_mode == 2:
            current_hist = HistoryBetweenTwoOuterResampling(M1=self.M1, M=self.M_resample)
            current_hist.add_new(res)
            self.histories_between_two_outer_resampling.append(current_hist)
        elif self.last_resampling_mode == 1:
            current_hist = self.histories_between_two_outer_resampling[-1]
            current_hist.add_new(res)
        else:
            assert self.last_resampling_mode == 0
            current_hist = self.histories_between_two_outer_resampling[-1]
            current_hist.update_last(res)
        return res

    def done(self, smc):
        res = super().done(smc)
        if res:
            assert np.allclose(smc.logLt, self.logging.weight_cumulator.logLT)
            assert np.allclose(smc.logLt, sum([hist.logLT for hist in self.histories_between_two_outer_resampling]))
            self.logging.inner_ESS_ratio.append(self._inner_ess_ratio(smc.wgts))
            self.logging.outer_ESS_ratio.append(ut.ESS_ratio(MultipleViewWeights(flattened_lw=smc.wgts.lw, M=self.M_resample, M1=self.M1).outer_weights))
            self.logging.resampling_mode.append(-1)
            self.logging.chain_length.append(0)
            self.logging.thinning.append(0)
            self.logging.done = True
        return res

    def __repr__(self):
        res = ''
        for i, (inner_ess, outer_ess, resampling_decision, chain_length, thinning) in enumerate(ut.zip_with_assert(self.logging.inner_ESS_ratio, self.logging.outer_ESS_ratio, self.logging.resampling_mode, self.logging.chain_length, self.logging.thinning)):
            res += 'M{ip1}: inner_ess={inner_ess:.2f}, outer_ess={outer_ess:.2f}, resampling={resampling_decision}, chain_length={cl}, thinning={thinning}\n'.format(inner_ess=inner_ess, outer_ess=outer_ess, resampling_decision=self.logging.resampling_mode_code[resampling_decision] if resampling_decision > -1 else None, ip1=i+1, cl=chain_length, thinning=thinning)
        return res[:-1]

    def compute_variance_QT_via_Lee_and_Whiteley(self, phi: tp.Callable[[np.ndarray], np.ndarray], no_last_block:int=None) -> float:
        # Build the function phi_tilde corresponding to the underlying Feynman-Kac model
        # noinspection PyProtectedMember
        arg = self.logging.compact_particle_history._X_last
        G_tilde = np.exp(self.logging.weight_cumulator.current_weights - np.max(self.logging.weight_cumulator.current_weights))
        phi_ = phi(self.logging.weight_cumulator.current_particles)
        G_phi = G_tilde * phi_
        G_phi_sum_by_island = np.sum(MultipleViewArray(flattened_view=G_phi, M=self.M_resample, M1=self.M1).detailled_view, axis=(0,2))
        G_tilde_sum_by_island = np.sum(MultipleViewArray(flattened_view=G_tilde, M=self.M_resample, M1=self.M1).detailled_view, axis=(0,2))
        with np.errstate(invalid='ignore'):
            ret = G_phi_sum_by_island/G_tilde_sum_by_island
            assert ret.shape == (self.M2,)
            ret[np.isnan(ret)] = 0
        phi_tilde = ut.ManuallyCachedFunction(_dummy_function)
        phi_tilde.cache(arg, result=ret)
        # Assert that the function phi_tilde is correctly constructed
        # noinspection PyTypeChecker
        assert np.allclose(self.pf_debug_access.calculate_QT(phi), self.logging.compact_particle_history.calculate_QT(phi_tilde))
        # noinspection PyTypeChecker
        return self.logging.compact_particle_history.estimate_var_QT(phi_tilde, no_last_block=no_last_block)

    def compute_variance_logLT_via_Lee_and_Whiteley(self, no_last_block:int=None) -> float:
        return self.logging.compact_particle_history.estimate_var_logLT(no_last_block)

    _error_msg_var_est_via_MCMC = 'Variance calculation via MCMC methods is only available if no inner-only resampling is performed.'

    def compute_variance_logLT_via_MCMC(self, method:str) -> float:
        # tested
        if 1 in self.logging.resampling_mode:
            raise TypeError(self._error_msg_var_est_via_MCMC)
        res = 0
        for normalized_logG in self.logging.weight_cumulator.pre_normalize_weights + [self.logging.weight_cumulator.current_weights]:
            tilded_and_reshaped = np.exp(normalized_logG - np.max(normalized_logG)).reshape(len(normalized_logG)//self.M_resample, self.M_resample)
            res += MCMC_variance(tilded_and_reshaped, method) * len(normalized_logG) / np.sum(tilded_and_reshaped)**2
        return res

    def compute_variance_QT_via_MCMC(self, phi: tp.Callable[[np.ndarray], np.ndarray], method:str) -> float:
        # tested
        if 1 in self.logging.resampling_mode:
            raise TypeError(self._error_msg_var_est_via_MCMC)
        G_tilde = np.exp(self.logging.weight_cumulator.current_weights - np.max(self.logging.weight_cumulator.current_weights))
        phi_ = phi(self.logging.weight_cumulator.current_particles)
        N = len(self.logging.weight_cumulator.current_particles)
        mu = np.mean(G_tilde * phi_)/np.mean(G_tilde)
        phi_bar = phi_ - mu
        G_bar_phi_bar_reshaped = (G_tilde * phi_bar).reshape(N//self.M_resample, self.M_resample)
        return 1/N * MCMC_variance(G_bar_phi_bar_reshaped, method)/np.mean(G_tilde)**2

# noinspection PyUnusedLocal
def _dummy_function(*args, **kwargs):
    raise AssertionError

class AdaptiveWastelessSMC(WastelessSMC):
    # tested
    def __init__(self, model: SMCSamplerModel, M:int, M1:int, max_N_particles: int, method_for_choosing_MCMC_length: str, ESSrmin_inner: float=0.5, ESSrmin_outer:float=0.5, inner_resampling_scheme: str='systematic', outer_resampling_scheme: str='multinomial', outer_essr_calculation_mode: str='standard'):
        super().__init__(model=model, k=None, M=M, M1=M1, ESSrmin_outer=ESSrmin_outer, ESSrmin_inner=ESSrmin_inner, inner_resampling_scheme=inner_resampling_scheme, outer_resampling_scheme=outer_resampling_scheme, outer_essr_calculation_mode=outer_essr_calculation_mode)
        self.max_N_particles = max_N_particles
        self.method_for_choosing_MCMC_length = method_for_choosing_MCMC_length
        self.target_N = None

    def M0(self, N):
        res = super().M0(N)
        self.target_N = N
        return res

    def apply_MCMC_then_flatten(self, t:int, starting_points: np.ndarray, target_N: int, starting_lw:np.ndarray) -> np.ndarray:
        # We will not use the parameter `target_N`, instead, it will be read as an attribute of self.
        tg = TargetedReproducer(starting_points=starting_points, starting_W=rs.Weights(starting_lw).W, target_ess=self.target_N, kernel=lambda _x: self.model.MCMC(t, _x), f=lambda _x: self.model.diag_function_for_adaptive_MCMC(t, _x), method=self.method_for_choosing_MCMC_length, union_function=self.collector, max_N_particles=self.max_N_particles, verbose=self.verbose, forceful=True)
        res = tg.run()
        ut.memory_tracker_add(t)
        self.logging.thinning.append(tg.k)
        self.logging.chain_length.append(tg.kernel_call + 1)
        return res