#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io
import math
from matplotlib import cm

path = sys.path.append('../../Code/')
sys.path.append(path)

import fct_correlations as correl
import fct_facilities as fac
import fct_data as dat

path_plot = 'Plots_latents_map_summary/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

# Dataset

Nstimuli = 8
Nsessions = 5

# Cross-covariances

time_step = 80
time_step_forward = 40
delay_step = 1
max_delay = 80

delays = np.arange(- max_delay,max_delay + 1,delay_step)
Ndelays = len(delays)
ind_nodelay = np.where(delays == 0)[0][0]

# Averages

time_range_idx = [ [int(120/time_step_forward), int(360/time_step_forward)], \
                   [int(1960/time_step_forward), int(2440/time_step_forward)]  ]
Nepochs = len(time_range_idx)

tau_max_all = [8,8]
tau_min_all = [-8,-8]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Retrieve cross-covariances from unspecific (sum) and specific (diff) latents

path_data = 'Data_latents_map/'

ii_session = ii_stimulus = 0
cross_cov_sum = fac.Retrieve('cross_cov_sum_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
tau_values = fac.Retrieve('tau_values.p', path_data)
t_values = fac.Retrieve('t_values.p', path_data)

cross_corr_sum_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
cross_cov_sum_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
cross_corr_diff_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
cross_cov_diff_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))

# cross_corr_sum_shuffle_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
# cross_cov_sum_shuffle_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
# cross_corr_diff_shuffle_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))
# cross_cov_diff_shuffle_array = np.zeros(( Nsessions, Nstimuli, cross_cov_sum.shape[0], cross_cov_sum.shape[1] ))

for ii_session in range(Nsessions):
	for ii_stimulus in range(Nstimuli):

		cross_corr_sum_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_corr_sum_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		cross_cov_sum_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_cov_sum_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		cross_corr_diff_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_corr_diff_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		cross_cov_diff_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_cov_diff_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)

		# cross_corr_sum_shuffle_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_corr_sum_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		# cross_cov_sum_shuffle_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_cov_sum_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		# cross_corr_diff_shuffle_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_corr_diff_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		# cross_cov_diff_shuffle_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_cov_diff_shuffle_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)

# Retrieve cross-covariances from PLS latents and overlaps between PLS axes and unspecific and specific axes

path_data = 'Data_PLSlatents_map/'

ii_session = ii_stimulus = 0
cross_corr_ex = fac.Retrieve('cross_corr_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
tau_values = fac.Retrieve('tau_values.p', path_data)
t_values = fac.Retrieve('t_values.p', path_data)

cross_corr_array = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[0], cross_corr_ex.shape[1] ))
overlap_sumV1_array = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[0], cross_corr_ex.shape[1] ))
overlap_sumV2_array = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[0], cross_corr_ex.shape[1] ))
overlap_diffV1_array = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[0], cross_corr_ex.shape[1] ))
overlap_diffV2_array = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[0], cross_corr_ex.shape[1] ))

for ii_session in range(Nsessions):
	for ii_stimulus in range(Nstimuli):

		cross_corr_array[ii_session, ii_stimulus,:] = fac.Retrieve('cross_corr_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		overlap_sumV1_array[ii_session, ii_stimulus,:] = fac.Retrieve('overlap_sumV1_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		overlap_sumV2_array[ii_session, ii_stimulus,:] = fac.Retrieve('overlap_sumV2_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		overlap_diffV1_array[ii_session, ii_stimulus,:] = fac.Retrieve('overlap_diffV1_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data)
		overlap_diffV2_array[ii_session, ii_stimulus,:] = fac.Retrieve('overlap_diffV2_sess'+str(ii_session)+'_stim'+str(ii_stimulus)+'.p', path_data) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute average cross-covariances, quantify amplitude and temporal structure

# Average cross-covariances over evoked and spontaneous windows

cov_sum_mean_time = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[1], Nepochs ))
cov_diff_mean_time = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[1], Nepochs ))
cov_PLS_mean_time = np.zeros(( Nsessions, Nstimuli, cross_corr_ex.shape[1], Nepochs ))

for ii_epoch in range(Nepochs):

    cov_sum_mean_time[:,:,:,ii_epoch] = np.mean(cross_cov_sum_array[:,:,time_range_idx[ii_epoch][0]:time_range_idx[ii_epoch][1],:], axis = 2)
    cov_diff_mean_time[:,:,:,ii_epoch] = np.mean(cross_cov_diff_array[:,:,time_range_idx[ii_epoch][0]:time_range_idx[ii_epoch][1],:], axis = 2)
    cov_PLS_mean_time[:,:,:,ii_epoch] = np.mean(cross_corr_array[:,:,time_range_idx[ii_epoch][0]:time_range_idx[ii_epoch][1],:], axis = 2)

# Average cross-covariances over stimuli, for each session

cov_sum_mean_ori = np.mean(cov_sum_mean_time, axis = 1)
cov_sum_sem_ori = np.std(np.mean(cov_sum_mean_time, axis = 1))/np.sqrt(Nstimuli)  

cov_diff_mean_ori = np.mean(cov_diff_mean_time, axis = 1)
cov_diff_sem_ori = np.std(cov_diff_mean_time, axis = 1)/np.sqrt(Nstimuli)  

cov_PLS_mean_ori = np.mean(cov_PLS_mean_time, axis = 1)
cov_PLS_sem_ori = np.std(cov_PLS_mean_time, axis = 1)/np.sqrt(Nstimuli)  

# Average cross-covariances over stimuli and sessions

cov_sum_mean_all = np.mean(cov_sum_mean_time, axis = (0,1))
cov_sum_sem_all = np.std(cov_sum_mean_time, axis = (0,1))/np.sqrt(Nsessions * Nstimuli) 

cov_diff_mean_all = np.mean(cov_diff_mean_time, axis = (0,1))
cov_diff_sem_all = np.std(cov_diff_mean_time, axis = (0,1))/np.sqrt(Nsessions * Nstimuli) 

cov_PLS_mean_all = np.mean(cov_PLS_mean_time, axis = (0,1))
cov_PLS_sem_all = np.std(cov_PLS_mean_time, axis = (0,1))/np.sqrt(Nsessions * Nstimuli)   

# Compute asymmetry scores

asym_scores_sum = np.zeros(( Nsessions, Nepochs ))
asym_scores_diff = np.zeros(( Nsessions, Nepochs ))
asym_scores_PLS = np.zeros(( Nsessions, Nepochs ))

for ii_session in range(Nsessions):
    for ii_epoch in range(Nepochs):

        tau_max = tau_max_all[ii_epoch]
        tau_min = tau_min_all[ii_epoch]

        tau_max_ind = np.where(delays == tau_max)[0][0] + 1
        tau_min_ind = np.where(delays == tau_min)[0][0]
        delays_window = np.arange(tau_min,tau_max + 1)
        n_delays_window = len(delays_window)

        asym_scores_sum[ii_session,ii_epoch] = dat.AsymmetryScore(cov_sum_mean_ori[ii_session, tau_min_ind:tau_max_ind,ii_epoch][None,None,:], delays_window)[0][0]
        asym_scores_diff[ii_session,ii_epoch] = dat.AsymmetryScore(cov_diff_mean_ori[ii_session, tau_min_ind:tau_max_ind, ii_epoch][None,None,:], delays_window)[0][0]
        asym_scores_PLS[ii_session,ii_epoch] = dat.AsymmetryScore(cov_PLS_mean_ori[ii_session,tau_min_ind:tau_max_ind,ii_epoch][None,None,:], delays_window)[0][0]

# Compute amplitudes

amplitude_sum = np.zeros(( Nsessions, Nepochs ))
amplitude_diff = np.zeros(( Nsessions, Nepochs ))
amplitude_PLS = np.zeros(( Nsessions, Nepochs ))

for ii_session in range(Nsessions):
    for ii_epoch in range(Nepochs):

        tau_max = tau_max_all[ii_epoch]
        tau_min = tau_min_all[ii_epoch]

        tau_max_ind = np.where(delays == tau_max)[0][0] + 1
        tau_min_ind = np.where(delays == tau_min)[0][0]
        delays_window = np.arange(tau_min,tau_max + 1)
        n_delays_window = len(delays_window)

        amplitude_sum[ii_session,ii_epoch] = dat.Amplitude(cov_sum_mean_ori[ii_session, tau_min_ind:tau_max_ind,ii_epoch][None,None,:], delays_window)[0][0]
        amplitude_diff[ii_session,ii_epoch] = dat.Amplitude(cov_diff_mean_ori[ii_session, tau_min_ind:tau_max_ind, ii_epoch][None,None,:], delays_window)[0][0]
        amplitude_PLS[ii_session,ii_epoch] = dat.Amplitude(cov_PLS_mean_ori[ii_session,tau_min_ind:tau_max_ind,ii_epoch][None,None,:], delays_window)[0][0]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plots

fac.SetPlotParams()
colors = ['#FF3C00', '#9A0071']
colors_transparent = ['#FFC8B7', '#F2C9E7']


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Cross-covariance functions

axis_scale_factor = 1000

fac.SetPlotDim(1.6, 1.5)

# Early evoked
    
fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_PLS_mean_all[:,0], color = colors[0])
plt.fill_between(delays, axis_scale_factor * (cov_PLS_mean_all[:,0] - cov_PLS_sem_all[:,0]), axis_scale_factor * (cov_PLS_mean_all[:,0] + cov_PLS_sem_all[:,0]), color = colors_transparent[0])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])  
plt.ylim(1.2, 2.1)     
plt.yticks([1.2, 2.1])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_PLS_evoked.pdf')

fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_sum_mean_all[:,0], color = colors[0])
plt.fill_between(delays, axis_scale_factor * (cov_sum_mean_all[:,0] - cov_sum_sem_all[:,0]), axis_scale_factor * (cov_sum_mean_all[:,0] + cov_sum_sem_all[:,0]), color = colors_transparent[0])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])  
plt.ylim(0.2, 0.8)          
plt.yticks([0.2, 0.8])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_unspecif_evoked.pdf')

fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_diff_mean_all[:,0], color = colors[0])
plt.fill_between(delays, axis_scale_factor * (cov_diff_mean_all[:,0] - cov_diff_sem_all[:,0]), axis_scale_factor * (cov_diff_mean_all[:,0] + cov_diff_sem_all[:,0]), color = colors_transparent[0])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])  
plt.ylim(0, 0.1)          
plt.yticks([0, 0.1])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_specif_evoked.pdf')

# Spontaneous

fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_PLS_mean_all[:,1], color = colors[1])
plt.fill_between(delays, axis_scale_factor * (cov_PLS_mean_all[:,1] - cov_PLS_sem_all[:,1]), axis_scale_factor * (cov_PLS_mean_all[:,1] + cov_PLS_sem_all[:,1]), color = colors_transparent[1])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])    
plt.ylim(0.6, 1.4)        
plt.yticks([0.6, 1.4])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_PLS_spont.pdf')

fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_sum_mean_all[:,1], color = colors[1])
plt.fill_between(delays, axis_scale_factor * (cov_sum_mean_all[:,1] - cov_sum_sem_all[:,1]), axis_scale_factor * (cov_sum_mean_all[:,1] + cov_sum_sem_all[:,1]), color = colors_transparent[1])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])   
plt.ylim(0.2, 0.8)         
plt.yticks([0.2, 0.8])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_unspecif_spont.pdf')

fig = plt.figure()

plt.plot(delays,axis_scale_factor * cov_diff_mean_all[:,1], color = colors[1])
plt.fill_between(delays, axis_scale_factor * (cov_diff_mean_all[:,1] - cov_diff_sem_all[:,1]), axis_scale_factor * (cov_diff_mean_all[:,1] + cov_diff_sem_all[:,1]), color = colors_transparent[1])
plt.axvline(x=0, linestyle = '-', color = 'k', linewidth=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(-max_delay, max_delay)
plt.xticks([-max_delay,0,max_delay])   
plt.ylim(0, 0.1)        
plt.yticks([0, 0.1])
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-covariance')

plt.savefig(path_plot+'cross_cov_specif_spont.pdf')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Overlap between CCA axis and unspecific/specific axis

fig = plt.figure()

plt.plot([0,1], [0,1], ls='--', color='0', linewidth=0.7)

for ii_time in range(len(time_range_idx)):

	for ii_session in range(Nsessions):

		plt.plot(np.nanmean(overlap_sumV1_array[ii_session,:,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
			np.nanmean(overlap_diffV1_array[ii_session,:,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
			'v', color='1', markersize=3.8, markeredgecolor=colors[ii_time], markeredgewidth='0.7')

		plt.plot( np.nanmean(overlap_sumV2_array[ii_session,:,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
			np.nanmean(overlap_diffV2_array[ii_session,:,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
			's', color='1', markersize=3.8, markeredgecolor=colors[ii_time], markeredgewidth='0.7')

        # for ii_stimulus in range(Nstimuli):

        #     plt.plot(np.nanmean(overlap_sumV1_array[ii_session,ii_stimulus,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
        #         np.nanmean(overlap_diffV1_array[ii_session,ii_stimulus,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
        #         'v', color='1', markersize=3.8, markeredgecolor=colors[ii_time], markeredgewidth='0.7')

        #     plt.plot( np.nanmean(overlap_sumV2_array[ii_session,ii_stimulus,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
        #         np.nanmean(overlap_diffV2_array[ii_session,ii_stimulus,time_range_idx[ii_time][0]:time_range_idx[ii_time][1]]), \
        #         's', color='1', markersize=3.8, markeredgecolor=colors[ii_time], markeredgewidth='0.7')

plt.xlabel('Unspecific')
plt.ylabel('Specific')

plt.xlim(0, 0.8)
plt.ylim(0, 0.8)

plt.xticks([0, 0.4, 0.8])
plt.yticks([0, 0.4, 0.8])

ax = plt.gca()
ax.set_aspect('equal', adjustable='box') 

plt.savefig(path_plot+'overlaps.pdf')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Asymmetry scores

fac.SetPlotDim(1.5, 1.5)

# PLS

fig = plt.figure()

np.random.seed(0) 
pos_early = np.ones(Nsessions) + np.random.normal(0, 0.1, size = Nsessions)
pos_spont = 2 * np.ones(Nsessions) + np.random.normal(0, 0.1, size = Nsessions)

plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(asym_scores_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [asym_scores_PLS[i, 0], asym_scores_PLS[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [np.mean(asym_scores_PLS[:,0]), np.mean(asym_scores_PLS[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, asym_scores_PLS[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(asym_scores_PLS[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, asym_scores_PLS[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(asym_scores_PLS[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Asymmetry score')
plt.yticks([-0.02, 0, 0.02])

plt.savefig(path_plot+'asym_scores_PLS.pdf') 

# Unspecific

fig = plt.figure()

plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(asym_scores_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [asym_scores_sum[i, 0], asym_scores_sum[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [np.mean(asym_scores_sum[:,0]), np.mean(asym_scores_sum[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, asym_scores_sum[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(asym_scores_sum[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, asym_scores_sum[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(asym_scores_sum[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Asymmetry score')
plt.yticks([-0.03, 0, 0.03, 0.06])

plt.savefig(path_plot+'asym_scores_unspecif.pdf')

# Specific

fig = plt.figure()

plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(asym_scores_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [asym_scores_diff[i, 0], asym_scores_diff[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [np.mean(asym_scores_diff[:,0]), np.mean(asym_scores_diff[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, asym_scores_diff[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(asym_scores_diff[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, asym_scores_diff[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(asym_scores_diff[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Asymmetry score')
plt.yticks([-0.55, 0, 0.55])

plt.savefig(path_plot+'asym_scores_specif.pdf')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Amplitudes

fac.SetPlotDim(1.35, 1.5)

# PLS

fig = plt.figure()

np.random.seed(0) 
pos_early = np.ones(Nsessions) + np.random.normal(0, 0.1, size = Nsessions)
pos_spont = 2 * np.ones(Nsessions) + np.random.normal(0, 0.1, size = Nsessions)

# plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(amplitude_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [axis_scale_factor * amplitude_PLS[i, 0], axis_scale_factor * amplitude_PLS[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [axis_scale_factor * np.mean(amplitude_PLS[:,0]), axis_scale_factor * np.mean(amplitude_PLS[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, axis_scale_factor * amplitude_PLS[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(axis_scale_factor * amplitude_PLS[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, axis_scale_factor * amplitude_PLS[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(axis_scale_factor * amplitude_PLS[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.yticks([0, 3.05])

plt.savefig(path_plot+'amplitude_PLS.pdf') 

# Unspecific

fig = plt.figure()

# plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(amplitude_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [axis_scale_factor * amplitude_sum[i, 0], axis_scale_factor * amplitude_sum[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [axis_scale_factor * np.mean(amplitude_sum[:,0]), axis_scale_factor * np.mean(amplitude_sum[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, axis_scale_factor * amplitude_sum[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(axis_scale_factor * amplitude_sum[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, axis_scale_factor * amplitude_sum[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(axis_scale_factor * amplitude_sum[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.yticks([0, 1.08])

plt.savefig(path_plot+'amplitude_unspecif.pdf')

# Specific

fig = plt.figure()

# plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(amplitude_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [axis_scale_factor * amplitude_diff[i, 0], axis_scale_factor * amplitude_diff[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [axis_scale_factor * np.mean(amplitude_diff[:,0]), axis_scale_factor * np.mean(amplitude_diff[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, axis_scale_factor * amplitude_diff[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, np.mean(axis_scale_factor * amplitude_diff[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, axis_scale_factor * amplitude_diff[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, np.mean(axis_scale_factor * amplitude_diff[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.3, 2.7)
plt.xticks([1,2], ['Evoked', 'Spont'])
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.yticks([0, 0.11])

plt.savefig(path_plot+'amplitude_specif.pdf')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Amplitudes

fac.SetPlotDim(1.5, 2.5)

# Stimulus-evoked

fig = plt.figure()

plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(asym_scores_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [axis_scale_factor * amplitude_sum[i, 0], axis_scale_factor * amplitude_diff[i, 0]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [np.mean(axis_scale_factor * amplitude_sum[:,0]), np.mean(axis_scale_factor * amplitude_diff[:,0])], color='0', linewidth=0.7)

plt.plot(pos_early, axis_scale_factor * amplitude_sum[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(1, axis_scale_factor * np.mean(amplitude_sum[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

plt.plot(pos_spont, axis_scale_factor * amplitude_diff[:,0], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[0], markeredgewidth = 0.7)
plt.plot(2, axis_scale_factor * np.mean(amplitude_diff[:,0]), 'o', markersize = 5, color = colors[0], markeredgecolor = colors[0], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.5, 2.5)
plt.xticks([1,2], ['Unspec', 'Spec'])
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.yticks([0, 1.1])

plt.savefig(path_plot+'amplitude_stim.pdf')

# Spontaneous

fig = plt.figure()

plt.axhline(y = 0, linestyle = '-', color = '0', linewidth=0.7)

# for i in range(asym_scores_PLS.shape[0]):
#     plt.plot([pos_early[i], pos_spont[i]], [axis_scale_factor * amplitude_sum[i, 1], axis_scale_factor * amplitude_diff[i, 1]], color='0.7', linewidth=0.7)
# plt.plot([1, 2], [axis_scale_factor * np.mean(amplitude_sum[:,1]), axis_scale_factor * np.mean(amplitude_diff[:,1])], color='0', linewidth=0.7)

plt.plot(pos_early, axis_scale_factor * amplitude_sum[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(1, axis_scale_factor * np.mean(amplitude_sum[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

plt.plot(pos_spont, axis_scale_factor * amplitude_diff[:,1], 'o', markersize = 3.8, color = 'white', markeredgecolor = colors[1], markeredgewidth = 0.7)
plt.plot(2, axis_scale_factor * np.mean(amplitude_diff[:,1]), 'o', markersize = 5, color = colors[1], markeredgecolor = colors[1], markeredgewidth = 1.1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim(0.5, 2.5)
plt.xticks([1,2], ['Unspec', 'Spec'])
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.yticks([0, 0.8])

plt.savefig(path_plot+'amplitude_spon.pdf')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 