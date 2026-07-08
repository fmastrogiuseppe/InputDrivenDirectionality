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

# Define population pair indices to evaluate cross-covariances

ind_E1 = 0
ind_E2 = 2


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

Nbins = 100
Cmax = 400
ylim = 0.4

fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Symmetric connectivity

case = 0
h = 0.5
params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_asymmetry = f"asymmetry_{'_'.join(param_strs)}.p"
filename_diff_E  = f"diff_E_{'_'.join(param_strs)}.p"
filename_diff_I  = f"diff_I_{'_'.join(param_strs)}.p"

asymmetry_all_inputs = fac.Retrieve(filename_asymmetry, path_data)
diff_E_all_inputs = fac.Retrieve(filename_diff_E, path_data)
diff_I_all_inputs = fac.Retrieve(filename_diff_I, path_data)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

sns.set_style("white")
sns.kdeplot(x = diff_E_all_inputs, y = asymmetry_all_inputs[ind_E1, ind_E2, :], cmap="Greys", fill=True,\
	levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2.)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Input diff to E')
plt.ylabel(r'Asymmetry')

filename_plot = f"density_plot_E_symm.pdf"
plt.savefig(path_plot + filename_plot) 

plt.show()


# 

fg = plt.figure()
ax = plt.axes(frameon=True)

sns.set_style("white")
sns.kdeplot(x = diff_I_all_inputs, y = asymmetry_all_inputs[ind_E1, ind_E2, :], cmap="Greys", fill=True,\
	levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2.)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Input diff to I')
plt.ylabel(r'Asymmetry')

filename_plot = f"density_plot_I_symm.pdf"
plt.savefig(path_plot + filename_plot) 

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Asymmetric connectivity

case = 1
hFF = 1.3
hFB = 0.2
params = [case, k, w, hFF, hFB]
params_labels = ['case', 'k', 'w', 'hFF', 'hFB']

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_asymmetry = f"asymmetry_{'_'.join(param_strs)}.p"
filename_diff_E  = f"diff_E_{'_'.join(param_strs)}.p"
filename_diff_I  = f"diff_I_{'_'.join(param_strs)}.p"

asymmetry_all_inputs_FF = fac.Retrieve(filename_asymmetry, path_data)
diff_E_all_inputs_FF = fac.Retrieve(filename_diff_E, path_data)
diff_I_all_inputs_FF = fac.Retrieve(filename_diff_I, path_data)

#

case = 1
hFF = 0.2
hFB = 1.3
params = [case, k, w, hFF, hFB]
params_labels = ['case', 'k', 'w', 'hFF', 'hFB']

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_asymmetry = f"asymmetry_{'_'.join(param_strs)}.p"
filename_diff_E  = f"diff_E_{'_'.join(param_strs)}.p"
filename_diff_I  = f"diff_I_{'_'.join(param_strs)}.p"

asymmetry_all_inputs_FB = fac.Retrieve(filename_asymmetry, path_data)
diff_E_all_inputs_FB = fac.Retrieve(filename_diff_E, path_data)
diff_I_all_inputs_FB = fac.Retrieve(filename_diff_I, path_data)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

sns.set_style("white")
sns.kdeplot(x = diff_E_all_inputs_FB, y = asymmetry_all_inputs_FB[ind_E1, ind_E2, :], cmap="Greens", fill=False, \
    levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2., alpha=1, linewidths=0.6)
sns.kdeplot(x = diff_E_all_inputs_FF, y = asymmetry_all_inputs_FF[ind_E1, ind_E2, :], cmap="Greys", fill=False, \
    levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2., alpha=1, linewidths=0.6)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Input diff to E')
plt.ylabel(r'Asymmetry')

filename_plot = f"density_plot_E_asymm.pdf"
plt.savefig(path_plot + filename_plot) 

plt.show()


# 


fg = plt.figure()
ax = plt.axes(frameon=True)

sns.set_style("white")
sns.kdeplot(x = diff_I_all_inputs_FB, y = asymmetry_all_inputs_FB[ind_E1, ind_E2, :], cmap="Greens", fill=False, \
    levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2., alpha=1, linewidths=0.6)
sns.kdeplot(x = diff_I_all_inputs_FF, y = asymmetry_all_inputs_FF[ind_E1, ind_E2, :], cmap="Greys", fill=False, \
    levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], bw_adjust=2., alpha=1, linewidths=0.6)

plt.axvline(x = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)
plt.axhline(y = 0, color = '0', linewidth = 0.3, linestyle = '-', alpha=1)

plt.xlim(-1, 1)
plt.xticks([-1, 0, 1])

plt.ylim(-ylim, ylim)
plt.yticks([-ylim, 0, ylim])

plt.xlabel(r'Input diff to I')
plt.ylabel(r'Asymmetry')

filename_plot = f"density_plot_I_asymm.pdf"
plt.savefig(path_plot + filename_plot) 

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
