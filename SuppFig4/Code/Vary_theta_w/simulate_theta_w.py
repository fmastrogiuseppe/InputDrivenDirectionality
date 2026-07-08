
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

path_data = 'Data_simulate_theta_w/'
path_plot = 'Plots_simulate_theta_w/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 4	# Use the two-area model for simplicity, but set inter-area connections to zero
tau_m = 10
Ntau = 1001

k = 1.01
h = 0.

nan_value = np.nan


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

theta_values = np.linspace(0, 90, 401)

# Connectivity

w_values = np.linspace(1e-5, 3.2, 401)


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(theta_values), len(w_values) ))
	lag = np.zeros(( N, N, len(theta_values), len(w_values) ))

	variance = np.zeros(( N, len(theta_values), len(w_values) ))
	tau = np.zeros(( N, len(theta_values), len(w_values) ))

	nan = np.zeros(( N, len(theta_values), len(w_values) ))

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_theta, theta in enumerate(theta_values):

		print (ii_theta)

		u = np.array([ np.cos(theta/180*np.pi), np.sin(theta/180*np.pi), 0, 0 ])
		U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


		for ii_w, w in enumerate(w_values):

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
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			asymmetry[:,:,ii_theta,ii_w] = th.AsymmetryScore(cross.real, tau_values)
			lag[:,:,ii_theta,ii_w] = th.PrincipalLag(cross.real, tau_values)
			variance[:,ii_theta,ii_w] = np.max(cross[np.arange(N), np.arange(N),:], 1)
			tau[:,ii_theta,ii_w] = (int(Ntau/2) - var.WidthHalfHeight(cross_input[:,0]))*(2*tau_m/Ntau)

			if (len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0): 

				nan[:,ii_theta,ii_w] = 1

			else:

				nan[:,ii_theta,ii_w] = nan_value

	# Compute boundary

	boundary = np.zeros(len(w_values))

	for ii_w, h in enumerate(w_values):
		if len(np.where(np.isnan(nan[0,:,ii_w])==True)[0])>0:
			boundary[ii_w] = theta_values[np.min(np.where(np.isnan(nan[0,:,ii_w])==True)[0])]
		else:
			boundary[ii_w] = 10


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
	extent = (np.min(theta_values), np.max(theta_values), np.min(w_values), np.max(w_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(variance[0,:,:])), cmap=cmap)

plt.axvline(x=45, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, w_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta_values), np.max(theta_values))
plt.xticks([np.min(theta_values), 45, np.max(theta_values)])

plt.ylim(np.min(w_values), np.max(w_values))
plt.yticks([np.min(w_values), np.max(w_values)])

plt.xlabel(r'Input angle')
plt.ylabel(r'Connectivity')

# plt.title(r'Response amplitude')

plt.savefig(path_plot+'variance_E.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

plt.imshow(variance[1,:,:].T, origin='lower', \
	extent = (np.min(theta_values), np.max(theta_values), np.min(w_values), np.max(w_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(variance[1,:,:])), cmap=cmap)

plt.axvline(x=45, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, w_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta_values), np.max(theta_values))
plt.xticks([np.min(theta_values), 45, np.max(theta_values)])

plt.ylim(np.min(w_values), np.max(w_values))
plt.yticks([np.min(w_values), np.max(w_values)])

plt.xlabel(r'Input angle')
plt.ylabel(r'Connectivity')

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
	extent = (np.min(theta_values), np.max(theta_values), np.min(w_values), np.max(w_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -vmax, vmax = vmax, cmap=cmap)

plt.axvline(x=45, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, w_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta_values), np.max(theta_values))
plt.xticks([np.min(theta_values), 45, np.max(theta_values)])

plt.ylim(np.min(w_values), np.max(w_values))
plt.yticks([np.min(w_values), np.max(w_values)])

plt.xlabel(r'Input angle')
plt.ylabel(r'Connectivity')

# plt.title(r'Asymmetry score')

plt.savefig(path_plot+'asymmetry.pdf')


plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag[0,1,:,:].T, origin='lower', \
	extent = (np.min(theta_values), np.max(theta_values), np.min(w_values), np.max(w_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,1,:,:])), vmax = np.max(np.fabs(lag[0,1,:,:])), cmap=cmap)

plt.axvline(x=45, ls='--', color='0', linewidth=0.5)

plt.plot(boundary, w_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta_values), np.max(theta_values))
plt.xticks([np.min(theta_values), 45, np.max(theta_values)])

plt.ylim(np.min(w_values), np.max(w_values))
plt.yticks([np.min(w_values), np.max(w_values)])

plt.xlabel(r'Input angle')
plt.ylabel(r'Connectivity')

# plt.title(r'Principal lag'))

plt.savefig(path_plot+'lag.pdf')


plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
