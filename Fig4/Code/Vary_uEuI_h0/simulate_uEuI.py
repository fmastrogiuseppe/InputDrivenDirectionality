
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

sys.path.append('../../../Code/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_uEuI/'
path_plot = 'Plots_simulate_uEuI/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 4
tau_m = 10
Ntau = 1001

k = 1.01
w = 1.5
h = 0.

nan_value = np.nan


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

theta1_values = np.linspace(-(0.1-1e-5), 0.1, 601)
theta2_values = theta1_values


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(theta1_values), len(theta2_values) ))
	lag = np.zeros(( N, N, len(theta1_values), len(theta2_values) ))


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

	for ii_theta1, theta1 in enumerate(theta1_values):

		print (ii_theta1)
		
		for ii_theta2, theta2 in enumerate(theta2_values):

			u = np.array([ 1+theta1/2., 1+theta2/2., 1-theta1/2., 1-theta2/2. ])

			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			if len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0:
				# and len(np.where(cross_input[1,0,:]<0)[0])==0 and len(np.where(cross_input[3,0,:]<0)[0])==0\
				# and len(np.where(cross[0,1,:]<0)[0])==0 and len(np.where(cross[2,3,:]<0)[0])==0: 

				asymmetry[:,:,ii_theta1,ii_theta2] = th.AsymmetryScore(cross.real, tau_values)
				lag[:,:,ii_theta1,ii_theta2] = th.PrincipalLag(cross.real, tau_values)

			else:

				asymmetry[:,:,ii_theta1,ii_theta2] = nan_value
				lag[:,:,ii_theta1,ii_theta2] = nan_value

	# Compute boundary

	boundary1 = np.zeros(len(theta2_values))
	boundary2 = np.zeros(len(theta2_values))

	for ii_theta2, theta2 in enumerate(theta2_values):

		if len(np.where(np.isnan(asymmetry[0,0,:int(len(theta1_values)/2),ii_theta2])==True)[0])>0:
			boundary1[ii_theta2] = theta1_values[np.max(np.where(np.isnan(asymmetry[0,0,:int(len(theta1_values)/2),ii_theta2])==True)[0])]
		else:
			boundary1[ii_theta2] = nan_value

		if len(np.where(np.isnan(asymmetry[0,0,int(len(theta1_values)/2):,ii_theta2])==True)[0])>0:
			boundary2[ii_theta2] = theta1_values[np.min(np.where(np.isnan(asymmetry[0,0,int(len(theta1_values)/2):,ii_theta2])==True)[0]+int(len(theta1_values)/2))]
		else:
			boundary2[ii_theta2] = nan_value


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(asymmetry, 'asymmetry.p', path_data)
	fac.Store(lag, 'lag.p', path_data)

	fac.Store(boundary1, 'boundary1.p', path_data)
	fac.Store(boundary2, 'boundary2.p', path_data)

else:

	asymmetry = fac.Retrieve('asymmetry.p', path_data)
	lag = fac.Retrieve('lag.p', path_data)

	boundary1 = fac.Retrieve('boundary1.p', path_data)
	boundary2 = fac.Retrieve('boundary2.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.7, 1.5)

#

vtop = 0.15

cmap = cm.get_cmap('RdGy_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry[0,2,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.5)
plt.axhline(y=0, ls='--', color='0', linewidth=0.5)
# plt.plot([np.min(theta1_values), np.max(theta1_values)], [np.min(theta2_values), np.max(theta2_values)], ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), 0, np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), 0, np.max(theta2_values)])

plt.xlabel(r'Input diff to E')
plt.ylabel(r'Input diff to I')

# plt.title(r'Asymmetry score')

plt.savefig(path_plot+'asymmetry.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag[0,2,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.5)
plt.axhline(y=0, ls='--', color='0', linewidth=0.5)
# plt.plot([np.min(theta1_values), np.min(theta2_values)], [np.max(theta1_values), np.max(theta2_values)], ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), 0, np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), 0, np.max(theta2_values)])

plt.xlabel(r'Input diff to E')
plt.ylabel(r'Input diff to I')

# plt.title(r'Principal lag'))

plt.savefig(path_plot+'lag.pdf')


plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
