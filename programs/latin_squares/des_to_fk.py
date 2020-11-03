from programs.latin_squares.model import LatinSquaresProblem, MetropolisOnLatinSquares
from libs_new.adaptive_tempering_new import AdaptiveTempering
from programs.executor import smc_sampler_model_to_fk

def des_to_fk(des):
    static_model = LatinSquaresProblem(d=des.d)
    model = AdaptiveTempering(static_model=static_model, mcmc_engine=MetropolisOnLatinSquares, caching_mode='attribute_insertion')
    return smc_sampler_model_to_fk(des, model)