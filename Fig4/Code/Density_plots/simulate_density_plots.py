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
import seaborn as sns

sys.path.append('../../../Code/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_density_plots/'
path_plot = 'Plots_simulate_density_plots/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# Network connectivity

N = 4
k = 1.01
w = 1.5

hFF = 0.25
hFB = 0.75


case = 1

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

lambdas, P = np.linalg.eig(W)
P_inv = np.linalg.inv(P)


# Temporal parameters
    
tau_m = 10
N_tau = 1001


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
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

N_inputs = 20000

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

    U_all_inputs = np.zeros((N, N, N_inputs))
    inputs_all = np.zeros((N, N_inputs))

    # Outputs
    
    cov_all_inputs = np.zeros((N, N, N_tau, N_inputs))
    cov_io_all_inputs = np.zeros((N, N, N_tau, N_inputs))

    valid_inputs = 0
    N_input_vectors = N_inputs
        
    while valid_inputs < N_input_vectors:
        
        print(valid_inputs)
        
        # Draw random input from uniform distribution
        
        u = np.array(np.random.uniform(low = 0, high = 1, size = N))
        u = u.reshape((N,1))
                      
        # Define input matrix U
                
        U = np.zeros((N, N))
        
        U[:,0] = np.squeeze(u)
        
        # Compute activity covariances and activity-input covariances
                              
        tau_values, cov_bar, cov, cov_bar_io, cov_io  = th.ComputeCovarianceTheory(lambdas, P, P_inv, U, tau_m, N_tau)
           
        # Check if we were in a good input regime and, if so, save the results
            
        condE_1 = np.all(cov_io[ind_E1, 0, :] >= 0)
        condE_2 = np.all(cov_io[ind_E2, 0, :] >= 0)
 
        all_cond = [condE_1, condE_2]
        
        if np.all(all_cond):
            
            ind = valid_inputs
            
            U_all_inputs[:,:,ind] = U
            
            cov_all_inputs[:,:,:,ind] = cov
            
            cov_io_all_inputs[:,:,:,ind] = cov_io
            
            valid_inputs += 1
                   

    # Initializations
    
    asymmetry_all_inputs = np.zeros((N, N, N_inputs))
    diff_E_all_inputs = np.zeros(N_inputs)
    diff_I_all_inputs = np.zeros(N_inputs)
    
    for i in range(N_inputs):
        
        print(i)
        
        cov = cov_all_inputs[:,:,:,i]
        U = U_all_inputs[:,:,i]
        
        asymmetry = th.AsymmetryScore(cov.real, tau_values)
        diff_E = U[0,0] - U[2,0]
        diff_I = U[1,0] - U[3,0]
        
        asymmetry_all_inputs[:,:,i] = asymmetry
        diff_E_all_inputs[i] = diff_E
        diff_I_all_inputs[i] = diff_I


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE
    
    param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]
    
    filename_asymmetry = f"asymmetry_{'_'.join(param_strs)}.p"
    filename_diff_E  = f"diff_E_{'_'.join(param_strs)}.p"
    filename_diff_I  = f"diff_I_{'_'.join(param_strs)}.p"
    
    fac.Store(asymmetry_all_inputs, filename_asymmetry, path_data)
    fac.Store(diff_E_all_inputs, filename_diff_E, path_data)
    fac.Store(diff_I_all_inputs, filename_diff_I, path_data)

else:
    
    param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]
    
    filename_asymmetry = f"asymmetry_{'_'.join(param_strs)}.p"
    filename_diff_E  = f"diff_E_{'_'.join(param_strs)}.p"
    filename_diff_I  = f"diff_I_{'_'.join(param_strs)}.p"
    
    asymmetry_all_inputs = fac.Retrieve(filename_asymmetry, path_data)
    diff_E_all_inputs = fac.Retrieve(filename_diff_E, path_data)
    diff_I_all_inputs = fac.Retrieve(filename_diff_I, path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

Nbins = 100
Cmax = 400
ylim = 0.7

fac.SetPlotParams()
fac.SetPlotDim(1.65, 1.5)

fg = plt.figure()

sns.set_style("white")
sns.kdeplot(x = diff_E_all_inputs, y = asymmetry_all_inputs[ind_E1, ind_E2, :], cmap="Greys", fill=True, bw_adjust=2.)
# plt.hist2d(diffuE, asymmetry[:,1], bins = Nbins, cmin=0, cmax=Cmax, range=[[-1,1],[-ylim,ylim]], cmap=cmap)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Asymmetry')

plt.show()

filename_plot = f"density_plot_diff_E_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 

# 


fg = plt.figure()
ax = plt.axes(frameon=True)

sns.set_style("white")
sns.kdeplot(x = diff_I_all_inputs, y = asymmetry_all_inputs[ind_E1, ind_E2, :], cmap="Greys", fill=True, bw_adjust=2.)
# plt.hist2d(diffuE, asymmetry[:,1], bins = Nbins, cmin=0, cmax=Cmax, range=[[-1,1],[-ylim,ylim]], cmap=cmap)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Diff input to I')
plt.ylabel(r'Asymmetry')

plt.show()

filename_plot = f"density_plot_diff_I_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
