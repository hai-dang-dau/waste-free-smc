from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
from programs.executor import input_to_output
from programs.latin_squares.des_to_fk import des_to_fk
from programs.latin_squares.diag_quantities import diag_quantities
from libs_new.utils import shut_off_numba_warnings
import numpy as np

np.seterr(divide='raise', over='raise', invalid='raise')
shut_off_numba_warnings()

path = './programs/latin_squares/'

input_to_output(path, des_to_fk, diag_quantities)