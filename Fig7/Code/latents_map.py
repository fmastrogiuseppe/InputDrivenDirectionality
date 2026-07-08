
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io
import math
from matplotlib import cm
import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=1

path = sys.path.append('../../Code/')
sys.path.append(path)

import fct_data as dat
import fct_correlations as correl
import fct_facilities as fac

path_data = 'Data_latents_map/'


if __name__ == "__main__":

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Parameters

	# Dataset

	Nstimuli = 8
	Nsessions = 5
	Nprocesses = 10

	stimulus_pairs = [ [0,4], [1,5], [2,6], [3,7], \
					   [4,0], [5,1], [6,2], [7,3] ]
	
	# Cross-covariances

	time_step = 80
	time_step_forward = 40
	delay_step = 1
	max_delay = 80
	timebin = 2

	doCompute = 1

	# Compute cross-covariances from unspecific (sum) and specific (diff) latents (each stimulus separately, in parallel)
	
	if doCompute:

		semaphore = multiprocessing.Semaphore(Nprocesses)
		p = []

		for ii_session in range(Nsessions):
			for ii_stimulus in range(Nstimuli):

				print ('Session: ', ii_session, ', Stimulus: ', ii_stimulus)

				semaphore.acquire()
				process = multiprocessing.Process(target=correl.LatentsMap, args=(ii_session, ii_stimulus, Nstimuli, stimulus_pairs, \
					time_step, time_step_forward, delay_step, max_delay, timebin, path_data, semaphore))
				process.start()
				p.append(process)


		for process in p: # Wait until all processes are done
			process.join()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
