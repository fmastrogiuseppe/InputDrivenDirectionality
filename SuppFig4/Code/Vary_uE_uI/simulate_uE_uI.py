
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

path_data = 'Data_simulate_uE_uI/'
path_plot = 'Plots_simulate_uE_uI/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 4	# Use the two-area model for simplicity, but set inter-area connections to zero
tau_m = 10
Ntau = 1001

w = 1.5
k = 1.01
h = 0.

nan_value = np.nan


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

uE_values = np.linspace(1e-5, 2.2, 401)

# Connectivity

uI_values = np.linspace(1e-5, 2.2, 401)


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(uE_values), len(uI_values) ))
	lag = np.zeros(( N, N, len(uE_values), len(uI_values) ))

	variance = np.zeros(( N, len(uE_values), len(uI_values) ))
	tau = np.zeros(( N, len(uE_values), len(uI_values) ))

	nan = np.zeros(( N, len(uE_values), len(uI_values) ))

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

	for ii_uE, uE in enumerate(uE_values):

		print (ii_uE)

		for ii_uI, uI in enumerate(uI_values):

			u = np.array([ uE, uI, 0, 0 ])
			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			asymmetry[:,:,ii_uE,ii_uI] = th.AsymmetryScore(cross.real, tau_values)
			lag[:,:,ii_uE,ii_uI] = th.PrincipalLag(cross.real, tau_values)
			variance[:,ii_uE,ii_uI] = np.max(cross[np.arange(N), np.arange(N),:], 1)
			tau[:,ii_uE,ii_uI] = (int(Ntau/2) - var.WidthHalfHeight(cross_input[:,0]))*(2*tau_m/Ntau)

			if (len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0): 

				nan[:,ii_uE,ii_uI] = 1

			else:

				nan[:,ii_uE,ii_uI] = nan_value

	# Compute boundary

	boundary = np.zeros(len(uI_values))

	for ii_uI, h in enumerate(uI_values):
		if len(np.where(np.isnan(nan[0,:,ii_uI])==True)[0])>0:
			boundary[ii_uI] = uE_values[np.max(np.where(np.isnan(nan[0,:,ii_uI])==True)[0])]
		else:
			boundary[ii_uI] = 10


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(asymmetry, 'asymmetry.p', path_data)
	fac.Store(lag, 'lag.p', path_data)

	fac.Store(variance, 'variance.p', path_data)
	fac.Store(tau, 'tau.p', path_data)

	fac.Store(boundary, 'boundary.p', path_data)

else:

	asymmetry = fac.Retrieve('asymmetry.p', path_data)
	lag = fac.Retrieve('lag.p', path_data)

	variance = fac.Retrieve('variance.p', path_data)
	tau = fac.Retrieve('tau.p', path_data)

	boundary = fac.Retrieve('boundary.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.6, 1.5)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

plt.imshow(variance[0,:,:].T, origin='lower', \
	extent = (np.min(uE_values), np.max(uE_values), np.min(uI_values), np.max(uI_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(variance[0,:,:])), cmap='pink_r')
	interpolation = 'nearest', vmin = 0, vmax = 4.5, cmap=cmap)

plt.plot(boundary[boundary<10], uI_values[boundary<10], color='0', linewidth=0.7)

plt.plot(2, 1, 'o', color='0.4')
plt.plot(1, 0.02, 'o', color='0.4')
plt.plot([1,2], [0.02,1], color='0.4', linewidth=0.5, ls='--')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE_values), np.max(uE_values))
plt.xticks([np.min(uE_values), np.max(uE_values)])

plt.ylim(np.min(uI_values), np.max(uI_values))
plt.yticks([np.min(uI_values), np.max(uI_values)])

plt.xlabel(r'Input to E')
plt.ylabel(r'Input to I')

# plt.title(r'Response amplitude')

plt.savefig(path_plot+'variance_E.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

plt.imshow(variance[1,:,:].T, origin='lower', \
	extent = (np.min(uE_values), np.max(uE_values), np.min(uI_values), np.max(uI_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(variance[1,:,:])), cmap=cmap)

plt.plot(boundary[boundary<10], uI_values[boundary<10], color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE_values), np.max(uE_values))
plt.xticks([np.min(uE_values), np.max(uE_values)])

plt.ylim(np.min(uI_values), np.max(uI_values))
plt.yticks([np.min(uI_values), np.max(uI_values)])

plt.xlabel(r'Input to E')
plt.ylabel(r'Input to I')

# plt.title(r'Response amplitude')

plt.savefig(path_plot+'variance_I.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

vmax = 0.4

cmap = cm.get_cmap('RdGy_r')
cmap.set_bad(color='0.4')

plt.imshow(asymmetry[0,1,:,:].T, origin='lower', \
	extent = (np.min(uE_values), np.max(uE_values), np.min(uI_values), np.max(uI_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -vmax, vmax = vmax, cmap=cmap)

plt.plot(boundary[boundary<10], uI_values[boundary<10], color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE_values), np.max(uE_values))
plt.xticks([np.min(uE_values), np.max(uE_values)])

plt.ylim(np.min(uI_values), np.max(uI_values))
plt.yticks([np.min(uI_values), np.max(uI_values)])

plt.xlabel(r'Input to E')
plt.ylabel(r'Input to I')

# plt.title(r'Asymmetry score')

plt.savefig(path_plot+'asymmetry.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag[0,1,:,:].T, origin='lower', \
	extent = (np.min(uE_values), np.max(uE_values), np.min(uI_values), np.max(uI_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,1,:,:])), vmax = np.max(np.fabs(lag[0,1,:,:])), cmap=cmap)

plt.plot(boundary[boundary<10], uI_values[boundary<10], color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uE_values), np.max(uE_values))
plt.xticks([np.min(uE_values), np.max(uE_values)])

plt.ylim(np.min(uI_values), np.max(uI_values))
plt.yticks([np.min(uI_values), np.max(uI_values)])

plt.xlabel(r'Input to E')
plt.ylabel(r'Input to I')

# plt.title(r'Principal lag'))

plt.savefig(path_plot+'lag.pdf')


plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
