
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
import math
from sklearn.cross_decomposition import CCA, PLSCanonical
import sys

import fct_data as dat
import fct_correlations as correl
import fct_facilities as fac

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Correlation maps

def LatentsMap(ii_session, ii_stimulus, Nstimuli, stimulus_pairs, \
				  time_step=80, time_step_forward=40, delay_step=1, max_delay=80, timebin=0, \
				  path_data='', semaphore=[]):

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Basic transformations

	# Import V1 and V2 datasets

	dataV1, dataV2, stimID, trialID = dat.GetData(ii_session, 'All', '../../Code/')
	dataV1, dataV2 = dat.SortByStimulus(dataV1, dataV2, stimID)

	Nrepet = int(dataV1.shape[1]/Nstimuli)

	# Cut silent neurons

	dataV1 = dat.ExcludeSilentNeurons(dataV1)
	dataV2 = dat.ExcludeSilentNeurons(dataV2)

	# Compute tuning curves

	tuningV1 = np.zeros(( dataV1.shape[0], Nstimuli ))
	tuningV2 = np.zeros(( dataV2.shape[0], Nstimuli ))

	for ii_stimulus_ in range(Nstimuli):

		tuningV1[:,ii_stimulus_] = np.mean(dataV1[:,ii_stimulus_*Nrepet:(ii_stimulus_+1)*Nrepet,:dat.stim_len], (1,2))
		tuningV2[:,ii_stimulus_] = np.mean(dataV2[:,ii_stimulus_*Nrepet:(ii_stimulus_+1)*Nrepet,:dat.stim_len], (1,2))

	# Select pair of stimuli and corresponding data

	ii_stimulus_A, ii_stimulus_B = stimulus_pairs[ii_stimulus]

	dataV1 = dataV1[:,ii_stimulus_A*Nrepet:(ii_stimulus_A+1)*Nrepet,:]
	dataV2 = dataV2[:,ii_stimulus_A*Nrepet:(ii_stimulus_A+1)*Nrepet,:]

	# Compute residuals

	# Remove the mean across trials

	dataV1 += - np.mean(dataV1,1)[:,None,:]
	dataV2 += - np.mean(dataV2,1)[:,None,:]

	# Remove the mean over time within each trial

	dataV1 += -np.mean(dataV1, 2)[:,:,None]
	dataV2 += -np.mean(dataV2, 2)[:,:,None]   


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Compute tuning index

	tuning_idx_V1 = (tuningV1[:,ii_stimulus_A]-tuningV1[:,ii_stimulus_B]) / \
					(tuningV1[:,ii_stimulus_A]+tuningV1[:,ii_stimulus_B]+1e-6)
	tuning_idx_V2 = (tuningV2[:,ii_stimulus_A]-tuningV2[:,ii_stimulus_B]) / \
					(tuningV2[:,ii_stimulus_A]+tuningV2[:,ii_stimulus_B]+1e-6)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Compute latents

	# Define unspecific (sum) and specific (diff) axes

	# Unspecific

	V1_sum_axis = np.ones(dataV1.shape[0])/np.sqrt(dataV1.shape[0])
	V2_sum_axis = np.ones(dataV2.shape[0])/np.sqrt(dataV2.shape[0])

	# Specific

	V1_diff_axis = tuning_idx_V1
	V1_diff_axis -= np.mean(V1_diff_axis)
	V1_diff_axis /= np.linalg.norm(V1_diff_axis)

	V2_diff_axis = tuning_idx_V2
	V2_diff_axis -= np.mean(V2_diff_axis)
	V2_diff_axis /= np.linalg.norm(V2_diff_axis)

	# Compute unspecific and specific latents

	sumV1 = np.tensordot(V1_sum_axis, dataV1, axes=(0, 0))
	sumV2 = np.tensordot(V2_sum_axis, dataV2, axes=(0, 0))

	diffV1 = np.tensordot(V1_diff_axis, dataV1, axes=(0, 0))
	diffV2 = np.tensordot(V2_diff_axis, dataV2, axes=(0, 0)) 


	# # Alternative way of computing axes and latents

	# idx_tuned_V1 = np.sort(np.flipud(np.argsort(tuning_idx_V1))[:int(0.5*len(tuning_idx_V1))])
	# idx_untuned_V1 = np.setdiff1d(np.arange(dataV1.shape[0]), idx_tuned_V1)
	# idx_tuned_V2 = np.sort(np.flipud(np.argsort(tuning_idx_V2))[:int(0.5*len(tuning_idx_V2))])
	# idx_untuned_V2 = np.setdiff1d(np.arange(dataV2.shape[0]), idx_tuned_V2)

	# # Define subpopulation signals

	# dataV1_A = np.sum(dataV1[idx_tuned_V1,:], 0) 
	# dataV2_A = np.sum(dataV2[idx_tuned_V2,:], 0) 

	# dataV1_B = np.sum(dataV1[idx_untuned_V1,:], 0) 
	# dataV2_B = np.sum(dataV2[idx_untuned_V2,:], 0)

	# # Compute latents

	# sumV1 = (dataV1_A+dataV1_B)/np.sqrt(dataV1.shape[0])
	# sumV2 = (dataV2_A+dataV2_B)/np.sqrt(dataV2.shape[0])

	# diffV1 = (dataV1_A-dataV1_B)/np.sqrt(dataV1.shape[0])
	# diffV2 = (dataV2_A-dataV2_B)/np.sqrt(dataV2.shape[0])


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Compute correlations

	delays = np.arange(-max_delay, max_delay+1, delay_step)
	ind_init_all = np.arange(0, dataV1.shape[2], time_step_forward)
	shuffle_idx = Shuffle(dataV1.shape[1]) 	# Runs the analysis also for a single shuffled dataset

	# Initialize variables to store cross-covariances

	cross_corr_sum = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_cov_sum = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_corr_diff = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_cov_diff = np.nan*np.zeros(( len(ind_init_all), len(delays) ))

	cross_corr_sum_shuffle = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_cov_sum_shuffle = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_corr_diff_shuffle = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	cross_cov_diff_shuffle = np.nan*np.zeros(( len(ind_init_all), len(delays) ))

	# Loop through time and delay indices

	for ii_time in range(len(ind_init_all)):

		print(ii_time, '/', len(ind_init_all))

		for ii_delay, delay in enumerate(delays):

			ind_init_X = ind_init_all[ii_time]
			ind_end_X = int(ind_init_X + time_step)
			Xsum = sumV1[:,ind_init_X:ind_end_X]
			Xdiff = diffV1[:,ind_init_X:ind_end_X]

			ind_init_Y = int(ind_init_X + delay)
			ind_end_Y = int(ind_init_Y + time_step)
			Ysum = sumV2[:,ind_init_Y:ind_end_Y]
			Ydiff = diffV2[:,ind_init_Y:ind_end_Y]
			Ysum_shuffle = sumV2[shuffle_idx,ind_init_Y:ind_end_Y]
			Ydiff_shuffle = diffV2[shuffle_idx,ind_init_Y:ind_end_Y]
            
			if (ind_init_X) >=0 and (ind_init_Y) >= 0 and (ind_end_X) <= dataV1.shape[2] and (ind_end_Y) <= dataV1.shape[2]:

				# Bin latents over time

				if timebin>0:

					Xsum = dat.Bin(Xsum[None,:], timebin)[0,:]
					Ysum = dat.Bin(Ysum[None,:], timebin)[0,:]
					Xdiff = dat.Bin(Xdiff[None,:], timebin)[0,:]
					Ydiff = dat.Bin(Ydiff[None,:], timebin)[0,:]

					Ysum_shuffle = dat.Bin(Ysum_shuffle[None,:], timebin)[0,:]
					Ydiff_shuffle = dat.Bin(Ydiff_shuffle[None,:], timebin)[0,:]

				# Compute cross-covariances and cross-correlations

				cross_corr_sum[ii_time,ii_delay] = np.corrcoef(np.reshape(Xsum, (Xsum.shape[0]*Xsum.shape[1])), np.reshape(Ysum, (Ysum.shape[0]*Ysum.shape[1])))[0,1]
				cross_cov_sum[ii_time,ii_delay] = np.cov(np.reshape(Xsum, (Xsum.shape[0]*Xsum.shape[1])), np.reshape(Ysum, (Ysum.shape[0]*Ysum.shape[1])))[0,1]
				cross_corr_diff[ii_time,ii_delay] = np.corrcoef(np.reshape(Xdiff, (Xdiff.shape[0]*Xdiff.shape[1])), np.reshape(Ydiff, (Ydiff.shape[0]*Ydiff.shape[1])))[0,1]
				cross_cov_diff[ii_time,ii_delay] = np.cov(np.reshape(Xdiff, (Xdiff.shape[0]*Xdiff.shape[1])), np.reshape(Ydiff, (Ydiff.shape[0]*Ydiff.shape[1])))[0,1]

				cross_corr_sum_shuffle[ii_time,ii_delay] = np.corrcoef(np.reshape(Xsum, (Xsum.shape[0]*Xsum.shape[1])), np.reshape(Ysum_shuffle, (Ysum.shape[0]*Ysum.shape[1])))[0,1]
				cross_cov_sum_shuffle[ii_time,ii_delay] = np.cov(np.reshape(Xsum, (Xsum.shape[0]*Xsum.shape[1])), np.reshape(Ysum_shuffle, (Ysum.shape[0]*Ysum.shape[1])))[0,1]
				cross_corr_diff_shuffle[ii_time,ii_delay] = np.corrcoef(np.reshape(Xdiff, (Xdiff.shape[0]*Xdiff.shape[1])), np.reshape(Ydiff_shuffle, (Ydiff.shape[0]*Ydiff.shape[1])))[0,1]
				cross_cov_diff_shuffle[ii_time,ii_delay] = np.cov(np.reshape(Xdiff, (Xdiff.shape[0]*Xdiff.shape[1])), np.reshape(Ydiff_shuffle, (Ydiff.shape[0]*Ydiff.shape[1])))[0,1]
				
				print(cross_corr_sum[ii_time, ii_delay])

	tau_values = np.arange(-max_delay, max_delay+1)
	t_values = np.arange(0, dataV1.shape[2], time_step_forward)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Store cross-covariances and cross-correlations

	fac.Store(cross_corr_sum, 'cross_corr_sum_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_cov_sum, 'cross_cov_sum_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_corr_diff, 'cross_corr_diff_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_cov_diff, 'cross_cov_diff_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)

	fac.Store(cross_corr_sum_shuffle, 'cross_corr_sum_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_cov_sum_shuffle, 'cross_cov_sum_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_corr_diff_shuffle, 'cross_corr_diff_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(cross_cov_diff_shuffle, 'cross_cov_diff_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)

	if (ii_session==0 and ii_stimulus==0): fac.Store(tau_values, 'tau_values.p', path_data)
	if (ii_session==0 and ii_stimulus==0): fac.Store(t_values, 't_values.p', path_data)

	semaphore.release()


def CCAMap(ii_session, ii_stimulus, Nstimuli, stimulus_pairs, \
				  time_step=80, time_step_forward=40, delay_step=1, max_delay=80, timebin=0, \
				  fct='cov', path_data='', semaphore=[]):

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Basic transformations

	# Import V1 and V2 datasets

	dataV1, dataV2, stimID, trialID = dat.GetData(ii_session, 'All', '../../Code/')
	dataV1, dataV2 = dat.SortByStimulus(dataV1, dataV2, stimID)

	Nrepet = int(dataV1.shape[1]/Nstimuli)

	# Cut silent neurons

	dataV1 = dat.ExcludeSilentNeurons(dataV1)
	dataV2 = dat.ExcludeSilentNeurons(dataV2)

	# Compute tuning curves

	tuningV1 = np.zeros(( dataV1.shape[0], Nstimuli ))
	tuningV2 = np.zeros(( dataV2.shape[0], Nstimuli ))

	for ii_stimulus_ in range(Nstimuli):

		tuningV1[:,ii_stimulus_] = np.mean(dataV1[:,ii_stimulus_*Nrepet:(ii_stimulus_+1)*Nrepet,:dat.stim_len], (1,2))
		tuningV2[:,ii_stimulus_] = np.mean(dataV2[:,ii_stimulus_*Nrepet:(ii_stimulus_+1)*Nrepet,:dat.stim_len], (1,2))

	# Select pair of stimuli and corresponding data

	ii_stimulus_A, ii_stimulus_B = stimulus_pairs[ii_stimulus]

	dataV1 = dataV1[:,ii_stimulus_A*Nrepet:(ii_stimulus_A+1)*Nrepet,:]
	dataV2 = dataV2[:,ii_stimulus_A*Nrepet:(ii_stimulus_A+1)*Nrepet,:]

	# Compute residuals

	# Remove the mean across trials

	dataV1 += - np.mean(dataV1,1)[:,None,:]
	dataV2 += - np.mean(dataV2,1)[:,None,:]

	# Remove the mean over time within each trial

	dataV1 += -np.mean(dataV1, 2)[:,:,None]
	dataV2 += -np.mean(dataV2, 2)[:,:,None] 


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Compute tuning index

	tuning_idx_V1 = (tuningV1[:,ii_stimulus_A]-tuningV1[:,ii_stimulus_B]) / \
					(tuningV1[:,ii_stimulus_A]+tuningV1[:,ii_stimulus_B]+1e-6)
	tuning_idx_V2 = (tuningV2[:,ii_stimulus_A]-tuningV2[:,ii_stimulus_B]) / \
					(tuningV2[:,ii_stimulus_A]+tuningV2[:,ii_stimulus_B]+1e-6)

	# Define unspecific (sum) and specific (diff) axes

	# Unspecific

	V1_sum_axis = np.ones(dataV1.shape[0])/np.sqrt(dataV1.shape[0])
	V2_sum_axis = np.ones(dataV2.shape[0])/np.sqrt(dataV2.shape[0])

	# Specific

	V1_diff_axis = tuning_idx_V1
	V1_diff_axis -= np.mean(V1_diff_axis)
	V1_diff_axis /= np.linalg.norm(V1_diff_axis)

	V2_diff_axis = tuning_idx_V2
	V2_diff_axis -= np.mean(V2_diff_axis)
	V2_diff_axis /= np.linalg.norm(V2_diff_axis)

	# # Alternative way of computing axes

	# idx_tuned_V1 = np.sort(np.flipud(np.argsort(tuning_idx_V1))[:int(0.5*len(tuning_idx_V1))])
	# idx_untuned_V1 = np.setdiff1d(np.arange(dataV1.shape[0]), idx_tuned_V1)
	# idx_tuned_V2 = np.sort(np.flipud(np.argsort(tuning_idx_V2))[:int(0.5*len(tuning_idx_V2))])
	# idx_untuned_V2 = np.setdiff1d(np.arange(dataV2.shape[0]), idx_tuned_V2)

	# V1_sum_axis = np.ones(dataV1.shape[0])/np.sqrt(dataV1.shape[0])
	# V1_diff_axis = np.ones(dataV1.shape[0])/np.sqrt(dataV1.shape[0])
	# V1_diff_axis[idx_tuned_V1] *= -1.

	# V2_sum_axis = np.ones(dataV2.shape[0])/np.sqrt(dataV2.shape[0])
	# V2_diff_axis = np.ones(dataV2.shape[0])/np.sqrt(dataV2.shape[0])
	# V2_diff_axis[idx_tuned_V2] *= -1.


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Compute CCA

	delays = np.arange(-max_delay, max_delay+1, delay_step)
	ind_init_all = np.arange(0, dataV1.shape[2], time_step_forward)

	# Initialize variables to store cross-covariances and overlaps

	corr = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	overlap_sumV1 = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	overlap_sumV2 = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	overlap_diffV1 = np.nan*np.zeros(( len(ind_init_all), len(delays) ))
	overlap_diffV2 = np.nan*np.zeros(( len(ind_init_all), len(delays) ))

	# Loop through time and delay indices

	for ii_time in range(len(ind_init_all)):

		print(ii_time, '/', len(ind_init_all))

		for ii_delay, delay in enumerate(delays):

			ind_init_X = ind_init_all[ii_time]
			ind_end_X = int(ind_init_X + time_step)
			X = dataV1[:,:,ind_init_X:ind_end_X]

			ind_init_Y = int(ind_init_X + delay)
			ind_end_Y = int(ind_init_Y + time_step)
			Y = dataV2[:,:,ind_init_Y:ind_end_Y]


			if (ind_init_X) >=0 and (ind_init_Y) >= 0 and (ind_end_X) <= dataV1.shape[2] and (ind_end_Y) <= dataV1.shape[2]:

				# Bin latents over time

				if timebin>0:
					X = dat.Bin(X, timebin)
					Y = dat.Bin(Y, timebin)

				# Apply CCA

				X_CCA = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2])).T
				Y_CCA = np.reshape(Y,(Y.shape[0],Y.shape[1]*Y.shape[2])).T

				if fct=='corr': cca = CCA(n_components=1, scale=False, max_iter=1000)
				elif fct=='cov': cca = PLSCanonical(n_components=1, scale=False, max_iter=1000)

				cca.fit(X_CCA,Y_CCA)

				X_c, Y_c = cca.transform(X_CCA,Y_CCA)

				# Compute cross-covariances or cross-correlations

				if fct=='corr': corr[ii_time, ii_delay] = np.corrcoef(X_c, Y_c, rowvar=False)[0,1]
				elif fct=='cov': corr[ii_time, ii_delay] = np.cov(X_c, Y_c, rowvar=False)[0,1]

				print(corr[ii_time, ii_delay])

				# Compute overlaps between CCA axes and unspecific and specific axes

				overlap_sumV1[ii_time, ii_delay] = np.fabs(np.dot(cca.x_weights_[:,0], V1_sum_axis))
				overlap_sumV2[ii_time, ii_delay] = np.fabs(np.dot(cca.y_weights_[:,0], V2_sum_axis))
				overlap_diffV1[ii_time, ii_delay] = np.fabs(np.dot(cca.x_weights_[:,0], V1_diff_axis))
				overlap_diffV2[ii_time, ii_delay] = np.fabs(np.dot(cca.y_weights_[:,0], V2_diff_axis))


	tau_values = np.arange(-max_delay, max_delay+1)
	t_values = np.arange(0, dataV1.shape[2], time_step_forward)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### Store cross-covariances and overlaps

	fac.Store(corr, 'cross_corr_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(overlap_sumV1, 'overlap_sumV1_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(overlap_sumV2, 'overlap_sumV2_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(overlap_diffV1, 'overlap_diffV1_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
	fac.Store(overlap_diffV2, 'overlap_diffV2_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)

	if (ii_session==0 and ii_stimulus==0): fac.Store(tau_values, 'tau_values.p', path_data)
	if (ii_session==0 and ii_stimulus==0): fac.Store(t_values, 't_values.p', path_data)

	semaphore.release()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Correlation utilities
	
def Shuffle(N):
	return np.random.choice(N, N, replace=False)

