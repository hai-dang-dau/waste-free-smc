def limit_number_cores(k:int, mod):
    #tested
    """
    :param k: the number of cores that one wants numpy to use
    Limit number of cores that numpy uses.
    Usage: Type import os, then limit_number_cores(k, os) before importing numpy.
    """
    k_str = str(k)
    mod.environ["OMP_NUM_THREADS"] = k_str  # export OMP_NUM_THREADS=4
    mod.environ["OPENBLAS_NUM_THREADS"] = k_str  # export OPENBLAS_NUM_THREADS=4
    mod.environ["MKL_NUM_THREADS"] = k_str  # export MKL_NUM_THREADS=6
    mod.environ["VECLIB_MAXIMUM_THREADS"] = k_str  # export VECLIB_MAXIMUM_THREADS=4
    mod.environ["NUMEXPR_NUM_THREADS"] = k_str  # export NUMEXPR_NUM_THREADS=6