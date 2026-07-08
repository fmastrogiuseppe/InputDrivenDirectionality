
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

sys.path.append('../../../CodeFM/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_uE/'
path_plot = 'Plots_simulate_uE/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 4
tau_m = 10
Ntau = 1001

k = 1.01
w = 1.5
h = 0.5

nan_value = np.nan


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

uI = 1
uE1_values = np.linspace(1e-5, 2.2, 601)
uE2_values = np.linspace(1e-5, 2.2, 601)


doCompute = 1

if doCompute:
	asymmetry = np.zeros(( N, N, len(uE1_values), len(uE2_values) ))
	lag = np.zeros(( N, N, len(uE1_values), len(uE2_values) ))


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SETUP NETWORK

	# Connectivity

	W = np.array( [ [ w, -k*w, h, 0 ],\
					[ w, -k*w, h, 0 ],\
					[ h, 0, w, -k*w ],\
					[ h, 0, w, -k*w ] ] )

	lambdas, P = np.linalg.eig(W)
	Pinv = np.linalg.inv(P)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_uE1, uE1 in enumerate(uE1_values):

		print (ii_uE1)
		
		for ii_uE2, uE2 in enumerate(uE2_values):

			u = np.array([ uE1, uI, uE2, uI ])
			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			if len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0:
				# and len(np.where(cross_input[1,0,:]<0)[0])==0 and len(np.where(cross_input[3,0,:]<0)[0])==0\
				# and len(np.where(cross[0,1,:]<0)[0])==0 and len(np.where(cross[2,3,:]<0)[0])==0: 

				asymmetry[:,:,ii_uE1,ii_uE2] = th.AsymmetryScore(cross.real, tau_values)
				lag[:,:,ii_uE1,ii_uE2] = th.PrincipalLag(cross.real, tau_values)

			else:

				asymmetry[:,:,ii_uE1,ii_uE2] = nan_value
				lag[:,:,ii_uE1,ii_uE2] = nan_value

	# Compute boundary

	boundary = np.zeros(len(uE2_values))

	for ii_uE2, uE2 in enumerate(uE2_values):
		if len(np.where(np.isnan(asymmetry[0,0,:,ii_uE2])==True)[0])>0:
			boundary[ii_uE2] = uE1_values[np.max(np.where(np.isnan(asymmetry[0,0,:,ii_uE2])==True)[0])]
		else:
			boundary[ii_uE2] = nan_value


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(asymmetry, 'asymmetry.p', path_data)
	fac.Store(lag, 'lag.p', path_data)

	fac.Store(boundary, 'boundary.p', path_data)

else:

	asymmetry = fac.Retrieve('asymmetry.p', path_data)
	lag = fac.Retrieve('lag.p', path_data)

	boundary = fac.Retrieve('boundary.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.6, 1.5)

#

vtop = 0.4

cmap = cm.get_cmap('RdGy_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry[0,2,:,:].T, origin='lower', \
	extent = (np.min(uE1_values), np.max(uE1_values), np.min(uE2_values), np.max(uE2_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.plot([np.min(uE1_values), np.max(uE1_values)], [np.min(uE2_values), np.max(uE2_values)], ls='--', color='0', linewidth=0.5)

plt.plot(boundary, uE2_values, color='0', linewidth=0.7)

plt.plot(1.8, 0.4, 'D', markersize=3, markerfacecolor='None', markeredgecolor='#FFBAD2', markeredgewidth=0.7)

plt.plot(k/(1.+h/w), k/(1.+h/w), 'o', markersize=2, color='0')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE1_values), np.max(uE1_values))
plt.xticks([np.min(uE1_values), np.max(uE1_values)])

plt.ylim(np.min(uE2_values), np.max(uE2_values))
plt.yticks([np.min(uE2_values), np.max(uE2_values)])

plt.xlabel(r'Input to E area 1')
plt.ylabel(r'Input to E area 2')

plt.savefig(path_plot+'asymmetry.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag[0,2,:,:].T, origin='lower', \
	extent = (np.min(uE1_values), np.max(uE1_values), np.min(uE2_values), np.max(uE2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.plot([np.min(uE1_values), np.min(uE2_values)], [np.max(uE1_values), np.max(uE2_values)], ls='--', color='0', linewidth=0.5)

plt.plot(boundary, uE2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE1_values), np.max(uE1_values))
plt.xticks([np.min(uE1_values), np.max(uE1_values)])

plt.ylim(np.min(uE2_values), np.max(uE2_values))
plt.yticks([np.min(uE2_values), np.max(uE2_values)])

plt.xlabel(r'Input to E area 1')
plt.ylabel(r'Input to E area 2')

plt.savefig(path_plot+'lag.pdf')


plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
