from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
import numpy as np
from programs.executor import input_to_output
from programs.orthant_proba.des_to_fk import des_to_fk
from programs.orthant_proba.diag_quantities import diag_quantities
from libs_new import utils as ut

np.seterr(divide='raise', over='raise', invalid='raise')
ut.shut_off_scipy_warnings()
ut.shut_off_numba_warnings()

path = './programs/orthant_proba/'

input_to_output(path, des_to_fk, diag_quantities)