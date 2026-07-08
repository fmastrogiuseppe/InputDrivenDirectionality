
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
from sklearn.cross_decomposition import PLSRegression

sys.path.append('../../../Code/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_epsilonh_phi/'
path_plot = 'Plots_simulate_epsilonh_phi/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 8
tau_m = 8
Ntau = 1001

k = 1.01
w = 1.5
h = 0.5

epsilon_w = 0.99

nan_value = np.nan

# Latents rotation matrix

R = np.zeros((int(N/2.), N))
Nlatents = R.shape[0]

R[0:2,0] = R[2:4,4] = 1 / np.sqrt(2)
R[0,2] = R[2,6] = 1 / np.sqrt(2)
R[1,2] = R[3,6] = -1 / np.sqrt(2)

# Reduced latents rotation matrix (to be applied to within-area E data)

R_red = np.ones((2,2))
R_red[1,1] = -1
R_red /= np.sqrt(2)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Epsilon h

epsilon_h_values = np.linspace(1e-3, 1-1e-3, 401)
phi_values = np.linspace(1e-3, 1-1e-3, 401)


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(epsilon_h_values), len(phi_values) ))
	lag = np.zeros(( N, N, len(epsilon_h_values), len(phi_values) ))

	asymmetry_latent = np.zeros(( Nlatents, Nlatents, len(epsilon_h_values), len(phi_values) ))
	lag_latent = np.zeros(( Nlatents, Nlatents, len(epsilon_h_values), len(phi_values) ))

	variance_ratio = np.zeros(( 2, len(epsilon_h_values), len(phi_values) ))
	crosscovariance_ratio = np.zeros(( len(epsilon_h_values), len(phi_values) ))
	tau_ratio = np.zeros(( len(epsilon_h_values), len(phi_values) ))



	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_epsilon_h, epsilon_h in enumerate(epsilon_h_values):

		print (ii_epsilon_h)
	
		for ii_phi, phi in enumerate(phi_values):


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### SETUP NETWORK

			# Connectivity

			W = np.array( [ [ (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_h)*h, 0, (1-epsilon_h)*h, 0 ],\
							[ (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_h)*h, 0, (1-epsilon_h)*h, 0 ],\
							[ (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_h)*h, 0, (1+epsilon_h)*h, 0 ],\
							[ (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_h)*h, 0, (1+epsilon_h)*h, 0 ],\
							[ (1+epsilon_h)*h, 0, (1-epsilon_h)*h, 0, (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_w)*w, -k*(1-epsilon_w)*w ],\
							[ (1+epsilon_h)*h, 0, (1-epsilon_h)*h, 0, (1+epsilon_w)*w, -k*(1+epsilon_w)*w, (1-epsilon_w)*w, -k*(1-epsilon_w)*w ],\
							[ (1-epsilon_h)*h, 0, (1+epsilon_h)*h, 0, (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_w)*w, -k*(1+epsilon_w)*w ],\
							[ (1-epsilon_h)*h, 0, (1+epsilon_h)*h, 0, (1-epsilon_w)*w, -k*(1-epsilon_w)*w, (1+epsilon_w)*w, -k*(1+epsilon_w)*w ] ] ) / 2

			lambdas, P = np.linalg.eig(W)
			P = P[:,np.flipud(np.argsort(lambdas))]
			lambdas = lambdas[np.flipud(np.argsort(lambdas))]

			Pinv = np.linalg.inv(P)

			# print(lambdas)


			# Inputs

			uI = 1
			uE1 = 3.2
			uE2 = 0.8

			u = 0.5 * np.array([ (1+phi)*uE1, (1+phi)*uI, (1-phi)*uE1, (1-phi)*uI, (1+phi)*uE2, (1+phi)*uI, (1-phi)*uE2, (1-phi)*uI ])
			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, cross_bar, cross, crossinput_bar, cross_input = \
				th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)

			cross_latent = th.RotateCovarianceTheory(cross, R)

			if (len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0) \
				and (len(np.where(cross_input[4,0,:]<0)[0])==0 and len(np.where(cross_input[6,0,:]<0)[0])==0):

				asymmetry[:,:,ii_epsilon_h,ii_phi] = th.AsymmetryScore(cross.real, tau_values)
				lag[:,:,ii_epsilon_h,ii_phi] = th.PrincipalLag(cross.real, tau_values)

				asymmetry_latent[:,:,ii_epsilon_h,ii_phi] = th.AsymmetryScore(cross_latent.real, tau_values)
				lag_latent[:,:,ii_epsilon_h,ii_phi] = th.PrincipalLag(cross_latent.real, tau_values)

				variance_ratio[0,ii_epsilon_h,ii_phi] = np.max(cross_latent[1,1,:])/np.max(cross_latent[0,0,:])
				variance_ratio[1,ii_epsilon_h,ii_phi] = np.max(cross_latent[3,3,:])/np.max(cross_latent[2,2,:])
				crosscovariance_ratio[ii_epsilon_h,ii_phi] = np.max(cross_latent[1,3,:])/np.max(cross_latent[0,2,:])

			else:

				asymmetry[:,:,ii_epsilon_h,ii_phi] = nan_value
				lag[:,:,ii_epsilon_h,ii_phi] = nan_value

				asymmetry_latent[:,:,ii_epsilon_h,ii_phi] = nan_value
				lag_latent[:,:,ii_epsilon_h,ii_phi] = nan_value

				variance_ratio[:,ii_epsilon_h,ii_phi] = np.nan
				crosscovariance_ratio[ii_epsilon_h,ii_phi] = np.nan


	# Compute boundary

	boundary = np.zeros(len(phi_values))

	for ii_phi, epsilon_w in enumerate(phi_values):
		if len(np.where(np.isnan(crosscovariance_ratio[:,ii_phi])==True)[0])>0:
			boundary[ii_phi] = epsilon_h_values[np.min(np.where(np.isnan(crosscovariance_ratio[:,ii_phi])==True)[0])]
		else:
			boundary[ii_phi] = 10

	### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### SAVE

	fac.Store(asymmetry, 'asymmetry.p', path_data)
	fac.Store(lag, 'lag.p', path_data)

	fac.Store(asymmetry_latent, 'asymmetry_latent.p', path_data)
	fac.Store(lag_latent, 'lag_latent.p', path_data)

	fac.Store(variance_ratio, 'variance_ratio.p', path_data)
	fac.Store(crosscovariance_ratio, 'crosscovariance_ratio.p', path_data)

	fac.Store(boundary, 'boundary.p', path_data)

else:

	asymmetry = fac.Retrieve('asymmetry.p', path_data)
	lag = fac.Retrieve('lag.p', path_data)

	asymmetry_latent = fac.Retrieve('asymmetry_latent.p', path_data)
	lag_latent = fac.Retrieve('lag_latent.p', path_data)

	variance_ratio = fac.Retrieve('variance_ratio.p', path_data)
	crosscovariance_ratio = fac.Retrieve('crosscovariance_ratio.p', path_data)

	boundary = fac.Retrieve('boundary.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.6, 1.5)

#

vtop = 0.2

cmap = cm.get_cmap('RdGy_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry_latent[0,2,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'asymmetry_unspunsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag_latent[0,2,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'lag_unspunsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry_latent[1,3,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'asymmetry_spsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag_latent[1,3,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'lag_spsp.pdf')

plt.show()

#

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(variance_ratio[0,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1, cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'varianceratio_1.pdf')

plt.show()

print(np.nanmax(variance_ratio[0,:,:].T-1))

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(variance_ratio[1,:,:].T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1, cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'varianceratio_2.pdf')

plt.show()

print(np.nanmax(variance_ratio[1,:,:].T-1))

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(crosscovariance_ratio.T, origin='lower', \
	extent = (np.min(epsilon_h_values), np.max(epsilon_h_values), np.min(phi_values), np.max(phi_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1, cmap=cmap)

plt.plot(boundary, phi_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel(r'Across- specializ')
plt.ylabel(r'Input specializ')

plt.savefig(path_plot+'crosscovarianceratio.pdf')

plt.show()

print(np.nanmax(crosscovariance_ratio.T-1))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
