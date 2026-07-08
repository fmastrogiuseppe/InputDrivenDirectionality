#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.io
import math
import scipy.optimize
from matplotlib import cm
import os

sys.path.append('../../../CodeFM/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_violin_plots/'
path_plot = 'Plots_simulate_violin_plots/'

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# Network connectivity

# case 0 - baseline, params [k, w, h];
# case 1 - asymmetric FF and FB, params [k, w, hFF, hFB];
# case 2 - across-area difference in w, params [k, w1, w2, h];
# case 3 - difference in across-area projection to E and I, params [] 

N = 4
k = 1.01
w = 1.5
h = 0.6

# hFF = 1.3
# hFB = 0.2
# w1 = 1
# w2 = 2
alpha = 0.9


case = 3

if case == 0:
    
    params = [case, k, w, h]
    params_labels = ['case', 'k', 'w', 'h']

    W = np.array( [ [ w, -k*w, h, 0 ],\
				[ w, -k*w, h, 0 ],\
				[ h, 0, w, -k*w ],\
				[ h, 0, w, -k*w ] ] )
    
if case == 1:
    
    params = [case, k, w, hFF, hFB]
    params_labels = ['case', 'k', 'w', 'hFF', 'hFB']
    
    W = np.array( [ [ w, -k*w, hFB, 0 ],\
				[ w, -k*w, hFB, 0 ],\
				[ hFF, 0, w, -k*w ],\
				[ hFF, 0, w, -k*w ] ] )
    

if case == 2:
    
    params = [case, k, w1, w2, h]
    params_labels = ['case', 'k', 'w1', 'w2', 'h']
    
    W = np.array( [ [ w1, -k*w1, h, 0 ],\
				[ w1, -k*w1, h, 0 ],\
				[ h, 0, w2, -k*w2 ],\
				[ h, 0, w2, -k*w2 ] ] )
    

if case == 3:
    
    params = [case, k, w, h, alpha]
    params_labels = ['case', 'k', 'w', 'h', 'alpha']
    
    W = np.array( [ [ w, -k*w, h, 0 ],\
				[ w, -k*w, alpha * h, 0 ],\
				[ h, 0, w, -k*w ],\
				[ alpha * h, 0, w, -k*w ] ] )
    
lambdas, P = np.linalg.eig(W)
P_inv = np.linalg.inv(P)


# Temporal parameters
    
tau_m = 10
N_tau = 1001

# Inputs

    # Define perturbation norm

N_deltas = 6
delta_min = 0
delta_max = 0.5
delta_range = delta_max - delta_min
delta_step = delta_range/(N_deltas - 1)
delta_list = np.arange(delta_min,delta_max + delta_step,delta_step)

    # Define perturbation angle

N_thetas = 10
theta_min = 0
theta_max = 2 
theta_range = theta_max - theta_min
theta_step = theta_range/(N_thetas)
theta_list = np.arange(theta_min, theta_max, theta_step)
thetas = np.pi * theta_list

    # Define number of random inputs to probe for each perturbation

N_inputs = 200


# Define population pair indices to evaluate cross-covariances

ind_E1 = 0
ind_E2 = 2

ind_pairs = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
ind_E1I1 = ind_pairs[0]
ind_E2I2 = ind_pairs[5]
ind_E_pair = 1

N_pairs = len(ind_pairs)
ind_deltazero = np.where(np.round(delta_list,2) == 0)[0][0]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

doCompute = 1

if doCompute:
    
    # Initialization
    
    # Parameterized quantities

    U_I_all_inputs = np.zeros((N, N, N_deltas, N_thetas, N_inputs))
    U_E_all_inputs = np.zeros((N, N, N_deltas, N_thetas, N_inputs))

    # Outputs
    
    cov_I_all_inputs = np.zeros((N, N, N_tau, N_deltas, N_thetas, N_inputs))
    cov_E_all_inputs = np.zeros((N, N, N_tau, N_deltas, N_thetas, N_inputs))
    cov_io_E_all_inputs = np.zeros((N, N, N_tau, N_deltas, N_thetas, N_inputs))
    cov_io_I_all_inputs = np.zeros((N, N, N_tau, N_deltas, N_thetas, N_inputs))

    valid_inputs = 0
    N_input_vectors = N_inputs
        
    while valid_inputs < N_input_vectors:
        
        print(valid_inputs)
        
        # Draw random input from uniform distribution
        
        u = np.array(np.random.uniform(low = 0.5, high = 1, size = N))
        u = u.reshape((N,1))
        
        # Initialize temporary variables for each input
        
        U_I_all = np.zeros((N, N, N_deltas, N_thetas))
        U_E_all = np.zeros((N, N, N_deltas, N_thetas))

        cov_I_all = np.zeros((N, N, N_tau, N_deltas, N_thetas))
        cov_E_all = np.zeros((N, N, N_tau, N_deltas, N_thetas))
        cov_io_E_all = np.zeros((N, N, N_tau, N_deltas, N_thetas))
        cov_io_I_all = np.zeros((N, N, N_tau, N_deltas, N_thetas))

        for i in range(N_deltas):
                   
            # Select perturbation size
            
            delta = delta_list[i]
            
            for m in range(N_thetas):
                
                # Select perturbation angle
                
                theta = thetas[m]
                
                delta_E1 = delta * np.cos(theta)
                delta_E2 = delta * np.sin(theta)
                
                delta_I1 = delta_E1
                delta_I2 = delta_E2
            
                # Define input
            
                u_probing_I = np.zeros((N, 1))
                u_probing_I[1] = delta_I1
                u_probing_I[3] = delta_I2
                
                u_probing_E = np.zeros((N, 1))
                u_probing_E[0] = delta_E1
                u_probing_E[2] = delta_E2
                
                u_I = u + u_probing_I
                u_E = u + u_probing_E
                
                # Define input matrix U
                        
                U_I = np.zeros((N, N))
                U_E = np.zeros((N, N))
                
                U_E[:,0] = np.squeeze(u_E)
                U_I[:,0] = np.squeeze(u_I)
                
                U_I_all[:,:,i,m] = U_I
                U_E_all[:,:,i,m] = U_E
                
                # tau_values, SigmaBar, Sigma, SigmaBarInput, SigmaInput
                
                # Compute activity covariances and activity-input covariances
                                      
                tau_values, cov_bar_I, cov_I, cov_bar_io_I, cov_io_I  = th.ComputeCovarianceTheory(lambdas, P, P_inv, U_I, tau_m, N_tau)
                tau_values, cov_bar_E, cov_E, cov_bar_io_E, cov_io_E  = th.ComputeCovarianceTheory(lambdas, P, P_inv, U_E, tau_m, N_tau)
                   
                cov_I_all[:,:,:,i,m] = cov_I
                cov_E_all[:,:,:,i,m] = cov_E
                
                cov_io_I_all[:,:,:,i,m] = cov_io_I
                cov_io_E_all[:,:,:,i,m] = cov_io_E
                

        # Check if we were in a good input regime and, if so, save the results
            
        condE_1 = np.all(cov_io_E_all[ind_E1, 0, :, :, :] >= 0)
        condE_2 = np.all(cov_io_E_all[ind_E2, 0, :, :, :] >= 0)
        condI_1 = np.all(cov_io_I_all[ind_E1, 0, :, :, :] >= 0)
        condI_2 = np.all(cov_io_I_all[ind_E2, 0, :, :, :] >= 0)
        
        all_cond = [condE_1, condE_2, condI_1, condI_2]
        
        if np.all(all_cond):
            
            ind = valid_inputs
            
            U_I_all_inputs[:,:,:,:,ind] = U_I_all
            U_E_all_inputs[:,:,:,:,ind] = U_E_all
            
            cov_I_all_inputs[:,:,:,:,:,ind] = cov_I_all
            cov_E_all_inputs[:,:,:,:,:,ind] = cov_E_all
            
            cov_io_I_all_inputs[:,:,:,:,:,ind] = cov_io_I_all
            cov_io_E_all_inputs[:,:,:,:,:,ind] = cov_io_E_all
            
            valid_inputs += 1
                   

    # Initializations
    
    asymmetry_E_all = np.zeros((N, N, N_deltas, N_thetas, N_inputs))
    asymmetry_I_all = np.zeros((N, N, N_deltas, N_thetas, N_inputs))
    delta_asymmetry_E_all = np.zeros((N_deltas, N_thetas, N_inputs))
    delta_asymmetry_I_all = np.zeros((N_deltas, N_thetas, N_inputs))
    
    
    for i in range(N_deltas):
        
        print(i)
        
        for m in range(N_thetas):
            
            for j in range(N_inputs):
                
                # Select covariance
                
                cov_E = cov_E_all_inputs[:, :, :, i, m, j]
                cov_I = cov_I_all_inputs[:, :, :, i, m, j]
               
                # Compute asymmetry scores
                
                asymmetry_E = th.AsymmetryScore(cov_E.real, tau_values)
                asymmetry_I = th.AsymmetryScore(cov_I.real, tau_values)                
                
                asymmetry_E_all[:,:,i,m,j] = asymmetry_E
                asymmetry_I_all[:,:,i,m,j] = asymmetry_I
                    
                # Compute difference in asymmetry scores between the unperturbed and perturbed input config.
    
                asymmetry_perturbed_E = asymmetry_E_all[ind_E1,ind_E2,i,m,j]
                asymmetry_unperturbed_E = asymmetry_E_all[ind_E1,ind_E2,ind_deltazero,m,j]
                delta_asymmetry_E_all[i,m,j] = np.abs(asymmetry_perturbed_E - asymmetry_unperturbed_E)
                
                asymmetry_perturbed_I = asymmetry_I_all[ind_E1,ind_E2,i,m,j]
                asymmetry_unperturbed_I = asymmetry_I_all[ind_E1,ind_E2,ind_deltazero,m,j]
                delta_asymmetry_I_all[i,m,j] = np.abs(asymmetry_perturbed_I - asymmetry_unperturbed_I)
                
    
    	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
    # SAVE
    
    param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]
    
    filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
    filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"
    
    fac.Store(delta_asymmetry_E_all, filename_E, path_data)
    fac.Store(delta_asymmetry_I_all, filename_I, path_data)


else:
    
    param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]
    
    filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
    filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"
    
    delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
    delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)
        

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT
                              
pos_1 = []
pos_2 = []
pos_1.append(1)
pos_2.append(2)

k = 1

while len(pos_1) < N_deltas:
    
    pos_1.append(pos_1[k-1] + 3)
    pos_2.append(pos_2[k-1] + 3)
    k = k + 1
    
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), pos_1, widths = 1, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), pos_2, widths = 1, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

colors = ['red', 'blue']  # Custom colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor('black')
    body.set_alpha(0.5)  # Optional transparency
    
for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor('black')
    body.set_alpha(0.5)  # Optional transparency
      
vp_1['cmedians'].set_color('darkred')      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color('darkblue')      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
tick_positions = (np.array(pos_1) + np.array(pos_2)) / 2
plt.xticks(tick_positions, np.round(delta_list, 2))
plt.yticks([0,0.5],[0,0.5])
plt.ylabel('$\Delta$ asymmetry')
plt.xlabel('Perturbation $\delta$')

filename_plot = f"violin_plot_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)


