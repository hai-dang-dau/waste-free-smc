from programs.orthant_proba.model import standardized_Gamma_and_a, TruncatedGaussianSimulator, OrthantProbability, TemperedOrthantProbability
import numpy as np
from programs.executor import smc_sampler_model_to_fk

def des_to_fk(des):
    Sigma = np.load(file='./programs/orthant_proba/data/Sigma_' + des.name + '.npy')
    a = np.load(file='./programs/orthant_proba/data/a_' + des.name + '.npy')
    Gamma, a = standardized_Gamma_and_a(Sigma,a)
    static_model = TruncatedGaussianSimulator()
    if not des.tempered:
        model = OrthantProbability(Gamma=Gamma, a=a, static_model=static_model)
    else:
        model = TemperedOrthantProbability(Gamma=Gamma, a=a, static_model=static_model)
    return smc_sampler_model_to_fk(des, model)