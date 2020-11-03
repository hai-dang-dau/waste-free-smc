All simulation results are stored in a 'out.csv' file. For example, simulation results for the binary regression problem are stored in the './programs/binary_regression/out.csv' file. There is also a 'in.csv' file, see explanations below.

Each line in the 'in.csv' file corresponds to a numerical experiment with specified parameters. For example, the first two lines of './programs/binary_regression/in.csv' are:

ESSrmin_inner,N,T,algo,caching_mode,cores,data,inner_resampling_scheme,k,max_N_particles,max_memory_in_MB,prior,processed,ESSrmin_outer,M,M1,outer_essr_calc_mode,outer_resampling_scheme
1.0,40000.0,100.0,ClassicalSMC,hash_table,50.0,sonar,multinomial,5.0,60000.0,4000.0,gaussian,True,,,,,

This means, in particular, that the first numerical experiment consists of running the Classical SMC algorithm with N = 40 000 and k = 5 on the sonar dataset using the multinomial resampling scheme. The algorithm will be run for T = 100 independent times using 50 cores.

The file 'out.csv' contains all the columns of 'in.csv', plus other columns indicating the result of the experiment. For example, there are 100 columns 'logLT0', ..., 'logLT99' corresponding to the produced estimates of the log normalizing constant logLT over the 100 runs.

Our results are properly seeded and so reproducible. We depend on the 'particles' package and we use its 04 July 2019 version, although it is expected to run on the actual version as well. To reproduce the file 'out.csv', delete the actual 'out.csv', re-create a new empty 'out.csv', replace the content of 'in.csv' by the content of 'in_reproduce.csv'. Finally, run 'main.py' (the working folder is './'). Warning: you may want to decrease the number of cores before starting.