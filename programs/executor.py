import pandas as pd
from pandas.errors import EmptyDataError
from typing import Callable, List
from libs_new.utils import OWA
import numpy as np
import datetime
import particles
from libs_new.cores_new import FeynmanKacNew, multiSMCNew, SMCTwice_signature, multiSMCTwice
from libs_new.smc_samplers_new import SMCSamplerModel, ClassicalSMCSampler, WastelessSMC, AdaptiveWastelessSMC
from functools import partial

def input_to_output(path:str, des_to_fk: Callable, diag_quantities: List):
    """
    Read the file `in.csv` (in the location defined by `path`) and execute jobs specified there.
    Outputs are written in `out.csv`
    :param des_to_fk: a function that turns a description of algorithm parameters to a FeynmanKac model
    :param diag_quantities: a list of diagnostic quantities (i.e. subclasses or objects of class programs.common_diagnostics.DiagQuantity) that will be used to produce diagnostics information to be written to the `out.csv` file.
    """
    jobs = pd.read_csv(path + 'in.csv')
    try:
        output = pd.read_csv(path + 'out.csv')
    except EmptyDataError:
        print('Output file is currently empty')
        output = pd.DataFrame()

    for i in range(len(jobs)):
        if not booleanize(jobs['processed'][i]):
            des = remove_float(jobs.iloc[i, :].to_dict())
            fk = des_to_fk(des)
            newline = dict(**jobs.iloc[i, :].to_dict(), **execute(fk=fk, des=des, diag_quantities=diag_quantities))
            output = output.append(newline, ignore_index=True)
            jobs.loc[i, 'processed'] = True
            output.to_csv(path + '/out.csv', index=False)
            jobs.to_csv(path + 'in.csv', index=False)

def smc_sampler_model_to_fk(des, model: SMCSamplerModel) -> FeynmanKacNew:
    if des.algo == 'ClassicalSMC':
        return ClassicalSMCSampler(model=model, k=des.k, ESSrmin=des.ESSrmin_inner, resampling_scheme=des.inner_resampling_scheme)
    if des.algo == 'WastelessSMC':
        return WastelessSMC(model=model, k=des.k, M=des.M, M1=des.M1, ESSrmin_outer=des.ESSrmin_outer, ESSrmin_inner=des.ESSrmin_inner, inner_resampling_scheme=des.inner_resampling_scheme, outer_resampling_scheme=des.outer_resampling_scheme, outer_essr_calculation_mode=des.outer_essr_calc_mode)
    if des.algo == 'AdaptiveWastelessSMC':
        return AdaptiveWastelessSMC(model=model, M=des.M, M1=des.M1, ESSrmin_outer=des.ESSrmin_outer, max_N_particles=des.max_N_particles, method_for_choosing_MCMC_length=des.method_MCMC_length, ESSrmin_inner=des.ESSrmin_inner, inner_resampling_scheme=des.inner_resampling_scheme, outer_resampling_scheme=des.outer_resampling_scheme, outer_essr_calculation_mode=des.outer_essr_calc_mode)
    raise ValueError('Unknown algorithm.')

def remove_float(x:dict):
    """
    :param x: a dictionary
    :return: an object with attributes. Values are converted from float to int whenever possible
    """
    # tested
    y = x.copy()
    for k, v in y.items():
        if isinstance(v, float) and (not np.isinf(v)) and (not np.isnan(v)) and (v - int(v) == 0):
            y[k] = int(v)
    return OWA(**y)

def booleanize(x):
    if not isinstance(x, str):
        return x
    if x == 'False' or x == '0.0' or x == '0':
        return False
    return x

def execute(fk, des: OWA, diag_quantities: List, seed=1234) -> dict:
    #tested
    """
    Run a Feymann-Kac model fk for des.T times using des.cores cores.

    :return: A dictionary with keys like cpu0, cpu1, ..., cpu(T-1); no_steps0, no_steps1, ..., no_steps(T-1); posterior_mean0, posterior_mean1, ... posterior_mean(T-1);... tracing exactly what happened in each run. Furthermore, synthetical keys like var_of_posterior_mean_estimated are also included. These diagostic quantities are defined by the diag_quantities variable.
    """
    np.random.seed(seed)
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')
    print(des)
    print(datetime.datetime.now())
    return process_results(runner(fk, des, diag_quantities))

def process_results(x:dict) -> dict:
    """
    :param x: A dictionary with keys like 'results' whose value is an array of length T
    :return: A new dictionary with keys like result0, result1, ..., resultT-1
    """
    # tested
    y = x.copy()
    for k, v in [(k2, v2) for k2, v2 in y.items()]:
        try:
            for i, j in enumerate(v):
                y[k[:-1] + str(i)] = j
            y.pop(k)
        except TypeError:
            pass
    return y

def _runner_of(pf, diag_quantities):
    """
    Pickable helper for the function `runner`. See the code of the function `runner` for meaning.
    """
    res = dict()
    for qu in diag_quantities:
        res.update({qu.name(): qu.extract(pf)})
    return res

def runner(fk, des: OWA, diag_quantities: List, verbose=False) -> dict:
    """
    Run a Feymann-Kac model fk for des.T times using des.cores cores.

    :return: A dictionary with keys like cpu, posterior_mean, posterior_variance, no_of_intermediate_dists, etc... The corresponding value of each of these keys is a list of length T tracing what happened for each run. Furthermore, synthetical keys like variance_of_posterior_mean are also included.
    """
    #tested
    # def of(pf):
    #     res = dict()
    #     for qu in diag_quantities:
    #         res.update({qu.name(): qu.extract(pf)})
    #     return res
    of = partial(_runner_of, diag_quantities=diag_quantities)

    out = dict()
    if isinstance(fk, FeynmanKacNew):
        # noinspection PyTypeChecker
        multiSMC = multiSMCNew(nruns=des.T, nprocs=des.cores, out_func=of, fk=fk, N=des.N, verbose=verbose, max_memory_in_MB=des.max_memory_in_MB)
    elif isinstance(fk, particles.FeynmanKac):
        # noinspection PyTypeChecker
        multiSMC = particles.multiSMC(nruns=des.T, nprocs=des.cores, out_func=of, fk=fk, N=des.N, ESSrmin = 1.0)
    elif isinstance(fk, SMCTwice_signature):
        # noinspection PyProtectedMember
        multiSMC = multiSMCTwice(nruns=des.T, nprocs=des.cores, out_func=of, verbose=verbose, **fk._asdict())
    else:
        raise ValueError('Unknown type of fk.')
    for q in diag_quantities:
        out.update({q.name() + 's': np.array([d[q.name()] for d in multiSMC])})
    mean_cpu_time = np.mean(out['cpus'])
    for q in diag_quantities:
        if not (q.exact(des) is None):
            if np.isnan(q.exact(des)):
                ref = np.mean(out[q.name() + 's'])
            else:
                ref = q.exact(des)
            SEs = (out[q.name() + 's'] - ref) ** 2 # squared errors
            mse_hat = np.mean(SEs)
            std_mse_hat = 1/np.sqrt(len(SEs)) * np.std(SEs)
            out.update({'adjusted_error_' + q.name() + '_low': mean_cpu_time * (mse_hat - 1.96 * std_mse_hat), 'adjusted_error_' + q.name() + '_high': mean_cpu_time * (mse_hat + 1.96 * std_mse_hat)})
            out['exact_' + q.name()] = q.exact(des)
    return out