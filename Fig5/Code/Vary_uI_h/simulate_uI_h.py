
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

path_data = 'Data_simulate_uI_h/'
path_plot = 'Plots_simulate_uI_h/'


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

uE = 1
uI_values = np.linspace(0, 1.2, 401)

# Connectivity

h_values = np.linspace(1e-5, 0.8, 401)


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(uI_values), len(h_values) ))
	lag = np.zeros(( N, N, len(uI_values), len(h_values) ))

	variance = np.zeros(( N, len(uI_values), len(h_values) ))
	tau = np.zeros(( N, len(uI_values), len(h_values) ))


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_uI, uI in enumerate(uI_values):

		print (ii_uI)
		
		for ii_h, h in enumerate(h_values):

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### SETUP NETWORK

			# Connectivity

			W = np.array( [ [ w, -k*w, h, 0 ],\
							[ w, -k*w, h, 0 ],\
							[ h, 0, w, -k*w ],\
							[ h, 0, w, -k*w ] ] )

			lambdas, P = np.linalg.eig(W)
			Pinv = np.linalg.inv(P)

			u = np.array([ uE, uI, 0.3*uE, 0 ])
			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			if (len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0): 

				asymmetry[:,:,ii_uI,ii_h] = th.AsymmetryScore(cross.real, tau_values)
				lag[:,:,ii_uI,ii_h] = th.PrincipalLag(cross.real, tau_values)
				variance[:,ii_uI,ii_h] = np.max(cross[np.arange(N), np.arange(N),:], 1)
				tau[:,ii_uI,ii_h] = var.WidthHalfHeight(cross_input[:,0])*(2*tau_m/Ntau)

			else:

				asymmetry[:,:,ii_uI,ii_h] = nan_value
				lag[:,:,ii_uI,ii_h] = nan_value
				variance[:,ii_uI,ii_h] = nan_value
				tau[:,ii_uI,ii_h] = nan_value

	# Compute boundary

	boundary = np.zeros(len(h_values))

	for ii_h, h in enumerate(h_values):
		if len(np.where(np.isnan(tau[0,:,ii_h])==True)[0])>0:
			boundary[ii_h] = uI_values[np.min(np.where(np.isnan(tau[0,:,ii_h])==True)[0])]
		else:
			boundary[ii_h] = 10


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
fac.SetPlotDim(1.55, 1.5)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

cmap = cm.get_cmap('RdBu_r')
cmap.set_bad(color='0.4')

plt.imshow(asymmetry[0,2,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.nanmax(np.fabs(asymmetry[0,2,:,:])), vmax = np.nanmax(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')

plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Asymmetry score')

plt.savefig(path_plot+'asymmetry.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag[0,2,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.nanmax(np.fabs(lag[0,2,:,:])), vmax = np.nanmax(np.fabs(lag[0,2,:,:])), cmap='RdBu_r')

plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Principal lag'))

plt.savefig(path_plot+'lag.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

plt.imshow(variance[0,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 0.85*np.nanmax(np.fabs(variance[0,:,:])), cmap='pink_r')

# plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Response amplitude')

plt.savefig(path_plot+'variance_firstarea.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(variance[2,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.nanmax(np.fabs(variance[2,:,:])), cmap='pink_r')

plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Response amplitude')

plt.savefig(path_plot+'variance_secondarea.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(tau[0,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 0.95*np.nanmax(np.fabs(tau[0,:,:])), cmap='pink_r')

# plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Response timescale')

plt.savefig(path_plot+'tau_firstarea.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(tau[2,:,:].T, origin='lower', \
	extent = (np.min(uI_values), np.max(uI_values), np.min(h_values), np.max(h_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.nanmax(np.fabs(tau[2,:,:])), cmap='pink_r')

plt.axvline(x=1, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, h_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(uI_values), np.max(uI_values))
plt.xticks([np.min(uI_values), np.max(uI_values)])

plt.ylim(np.min(h_values), np.max(h_values))
plt.yticks([np.min(h_values), np.max(h_values)])

plt.xlabel(r'Input to I')
plt.ylabel(r'Inter-area conn')

# plt.title(r'Response timescale')

plt.savefig(path_plot+'tau_secondarea.pdf')


plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
