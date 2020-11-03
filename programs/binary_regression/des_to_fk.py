from programs.binary_regression.model import BinaryRegression
from libs_new.adaptive_tempering_new import AdaptiveTempering
from libs_new.mcmc_new import RWMHv2
from libs_new.utils import NumpyHashWrapper
from programs.executor import smc_sampler_model_to_fk

def des_to_fk(des):
    static_model = BinaryRegression(data=des.data, prior=des.prior)
    model = AdaptiveTempering(static_model=static_model, mcmc_engine=RWMHv2, caching_mode=des.caching_mode, max_cache_size=des.max_N_particles, hash_and_eq_wrapper=NumpyHashWrapper)
    return smc_sampler_model_to_fk(des, model)