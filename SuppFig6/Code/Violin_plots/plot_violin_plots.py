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

sys.path.append('../../../CodeFM/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_violin_plots/'
path_plot = 'Plots_simulate_violin_plots/'

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# Inputs

N_deltas = 6
N_deltas = 6
delta_min = 0
delta_max = 0.5
delta_range = delta_max - delta_min
delta_step = delta_range/(N_deltas - 1)
delta_list = np.arange(delta_min,delta_max + delta_step,delta_step)

N_thetas = 10
N_inputs = 200

# Violin positions

pos_gap = 0.02

    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT BASELINE

case = 0
N = 4
k = 1.01
w = 1.5
h = 0.5

params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

colors = ['#F4B3C1', '#CCDBF2']  # Custom colors
colors_dark = ['#E94F4C', '#4C8CCA']

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 

plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])

plt.ylabel(r'$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_baseline_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT SMALLER H

case = 0
N = 4
k = 1.01
w = 1.5
h = 0.25

params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_smallh_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT LARGER H

case = 0
N = 4
k = 1.01
w = 1.5
h = 0.95

params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_largeh_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT LARGER K

case = 0
N = 4
k = 1.25
w = 1.5
h = 0.5

params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_largek_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT LARGER W

case = 0
N = 4
k = 1.01
w = 5
h = 0.5

params = [case, k, w, h]
params_labels = ['case', 'k', 'w', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_largew_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT DIFF W ACROSS AREAS

case = 2
N = 4
k = 1.01
w1 = 1
w2 = 2
h = 0.5

params = [case, k, w1, w2, h]
params_labels = ['case', 'k', 'w1', 'w2', 'h']


fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_diffw_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT DIFF H ACROSS AREAS

case = 1
N = 4
k = 1.01
w = 1.5
hFF = 1.3
hFB = 0.2

params = [case, k, w, hFF, hFB]
params_labels = ['case', 'k', 'w', 'hFF', 'hFB']

fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_diffh_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT DIFF H ACROSS E AND I

case = 3
N = 4
k = 1.01
w = 1.5
h = 0.6
alpha = 0.9
    
params = [case, k, w, h, alpha]
params_labels = ['case', 'k', 'w', 'h', 'alpha']

fac.SetPlotParams()
fac.SetPlotDim(1.5, 1.4)

param_strs = [f"{label}_{str(val).replace('.', 'p')}" for label, val in zip(params_labels, params)]

filename_E = f"violin_plot_E_{'_'.join(param_strs)}.p"
filename_I = f"violin_plot_I_{'_'.join(param_strs)}.p"

delta_asymmetry_E_all = fac.Retrieve(filename_E, path_data)
delta_asymmetry_I_all = fac.Retrieve(filename_I, path_data)

# 
    
fac.SetPlotParams()
fac.SetPlotDim(2, 1.65)

fg = plt.figure()

vp_1 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_E_all, (N_deltas, N_thetas * N_inputs)).T), delta_list-pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)
vp_2 = plt.violinplot(np.abs(np.reshape(delta_asymmetry_I_all, (N_deltas, N_thetas * N_inputs)).T), delta_list+pos_gap, widths = .035, showmeans = False, showextrema = False, showmedians = True)

# Change violin colors

for i, body in enumerate(vp_1['bodies']):
    body.set_facecolor(colors[0])
    body.set_edgecolor(colors_dark[0])
    body.set_alpha(1)  # Optional transparency

for i, body in enumerate(vp_2['bodies']):
    body.set_facecolor(colors[1])
    body.set_edgecolor(colors_dark[1])
    body.set_alpha(1)  # Optional transparency

vp_1['cmedians'].set_color(colors_dark[0])      
vp_1['cmedians'].set_linewidth(1)   
vp_2['cmedians'].set_color(colors_dark[1])      
vp_2['cmedians'].set_linewidth(1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xticks(delta_list)
plt.xlim(-0.06, 0.56)
plt.yticks([0, 0.5])
plt.ylim([-0.02, 0.5])
plt.ylabel('$|\Delta$ Asymmetry$|$')
plt.xlabel('Perturbation amplitude')

filename_plot = f"violin_plot_diffhEI_{'_'.join(param_strs)}.pdf"
plt.savefig(path_plot + filename_plot) 

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
