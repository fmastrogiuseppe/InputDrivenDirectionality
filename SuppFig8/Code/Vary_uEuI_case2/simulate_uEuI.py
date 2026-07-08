
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

N = 8
tau_m = 8
Ntau = 1001

k = 1.01
w = 1.5
h = 0.5

epsilon_w = 0.2
epsilon_h = 0.7
epsilon = epsilon_w

phi = 0.99

nan_value = np.nan


# Latents rotation matrix

R = np.zeros((int(N/2.), N))
Nlatents = R.shape[0]

R[0:2,0] = R[2:4,4] = 1 / np.sqrt(2)
R[0,2] = R[2,6] = 1 / np.sqrt(2)
R[1,2] = R[3,6] = -1 / np.sqrt(2)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

theta1_values = np.linspace(-0.3, 0.3, 401)
theta2_values = theta1_values


doCompute = 1

if doCompute:

	asymmetry = np.zeros(( N, N, len(theta1_values), len(theta2_values) ))
	lag = np.zeros(( N, N, len(theta1_values), len(theta2_values) ))

	asymmetry_latent = np.zeros(( Nlatents, Nlatents, len(theta1_values), len(theta2_values) ))
	lag_latent = np.zeros(( Nlatents, Nlatents, len(theta1_values), len(theta2_values) ))

	variance_ratio = np.zeros(( 2, len(theta1_values), len(theta2_values) ))
	crosscovariance_ratio = np.zeros(( len(theta1_values), len(theta2_values) ))


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


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_theta1, theta1 in enumerate(theta1_values):

		print (ii_theta1)
		
		for ii_theta2, theta2 in enumerate(theta2_values):

			u = 0.5 * np.array([ (1+phi)*(1+theta1/2.), (1+phi)*(1+theta2/2.), (1-phi)*(1+theta1/2.), (1-phi)*(1+theta2/2.), \
						   (1+phi)*(1-theta1/2.), (1+phi)*(1-theta2/2.), (1-phi)*(1-theta1/2.), (1-phi)*(1-theta2/2.) ])
			U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			#### THEORY

			tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=Ntau)
			cross_latent = th.RotateCovarianceTheory(cross, R)

			if (len(np.where(cross_input[0,0,:]<0)[0])==0 and len(np.where(cross_input[2,0,:]<0)[0])==0) \
				and (len(np.where(cross_input[4,0,:]<0)[0])==0 and len(np.where(cross_input[6,0,:]<0)[0])==0): 

				asymmetry[:,:,ii_theta1,ii_theta2] = th.AsymmetryScore(cross.real, tau_values)
				lag[:,:,ii_theta1,ii_theta2] = th.PrincipalLag(cross.real, tau_values)

				asymmetry_latent[:,:,ii_theta1,ii_theta2] = th.AsymmetryScore(cross_latent.real, tau_values)
				lag_latent[:,:,ii_theta1,ii_theta2] = th.PrincipalLag(cross_latent.real, tau_values)

				variance_ratio[0,ii_theta1,ii_theta2] = np.max(cross_latent[1, 1,:])/np.max(cross_latent[0, 0,:])
				variance_ratio[1,ii_theta1,ii_theta2] = np.max(cross_latent[3, 3,:])/np.max(cross_latent[2, 2,:])

				crosscovariance_ratio[ii_theta1,ii_theta2] = np.max(cross_latent[1, 3,:])/np.max(cross_latent[0, 2,:])

			else:

				asymmetry[:,:,ii_theta1,ii_theta2] = nan_value
				lag[:,:,ii_theta1,ii_theta2] = nan_value

				asymmetry_latent[:,:,ii_theta1,ii_theta2] = nan_value
				lag_latent[:,:,ii_theta1,ii_theta2] = nan_value

				variance_ratio[:,ii_theta1,ii_theta2] = np.nan
				crosscovariance_ratio[ii_theta1,ii_theta2] = np.nan


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

	fac.Store(asymmetry_latent, 'asymmetry_latent.p', path_data)
	fac.Store(lag_latent, 'lag_latent.p', path_data)

	fac.Store(variance_ratio, 'variance_ratio.p', path_data)
	fac.Store(crosscovariance_ratio, 'crosscovariance_ratio.p', path_data)

	fac.Store(boundary1, 'boundary1.p', path_data)
	fac.Store(boundary2, 'boundary2.p', path_data)

else:

	asymmetry = fac.Retrieve('asymmetry.p', path_data)
	lag = fac.Retrieve('lag.p', path_data)

	asymmetry_latent = fac.Retrieve('asymmetry_latent.p', path_data)
	lag_latent = fac.Retrieve('lag_latent.p', path_data)

	variance_ratio = fac.Retrieve('variance_ratio.p', path_data)
	crosscovariance_ratio = fac.Retrieve('crosscovariance_ratio.p', path_data)

	boundary1 = fac.Retrieve('boundary1.p', path_data)
	boundary2 = fac.Retrieve('boundary2.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.65, 1.5)

#

vtop = 0.08

cmap = cm.get_cmap('RdGy_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry_latent[0,2,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

plt.plot(2.8, 0.8, 'D', markersize=2, color='1', markeredgecolor='k', markeredgewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'asymmetry_unspunsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag_latent[0,2,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'lag_unspunsp.pdf')

plt.show()

#

vtop = 0.08

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(asymmetry_latent[1,3,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	# interpolation = 'nearest', vmin = -np.max(np.fabs(asymmetry[0,2,:,:])), vmax = np.max(np.fabs(asymmetry[0,2,:,:])), cmap='RdBu_r')
	interpolation = 'nearest', vmin=-vtop, vmax=vtop, cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'asymmetry_spsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(lag_latent[1,3,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(lag[0,2,:,:])), vmax = np.max(np.fabs(lag[0,2,:,:])), cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'lag_spsp.pdf')

plt.show()

#

cmap = cm.get_cmap('pink_r')
cmap.set_bad(color='0.4')

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(variance_ratio[0,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1, cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'varianceratio_1.pdf')

plt.show()

print(np.nanmin(variance_ratio[0,:,:].T-1))

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(variance_ratio[1,:,:].T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1, cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'varianceratio_2.pdf')

plt.show()

print(np.nanmin(variance_ratio[1,:,:].T-1))

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(crosscovariance_ratio.T, origin='lower', \
	extent = (np.min(theta1_values), np.max(theta1_values), np.min(theta2_values), np.max(theta2_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 1., cmap=cmap)

plt.axvline(x=0, ls='--', color='0', linewidth=0.7)
plt.axhline(y=0, ls='--', color='0', linewidth=0.7)

plt.plot(boundary1, theta2_values, color='0', linewidth=0.7)
plt.plot(boundary2, theta2_values, color='0', linewidth=0.7)

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(np.min(theta1_values), np.max(theta1_values))
plt.xticks([np.min(theta1_values), np.max(theta1_values)])

plt.ylim(np.min(theta2_values), np.max(theta2_values))
plt.yticks([np.min(theta2_values), np.max(theta2_values)])

plt.xlabel(r'Diff input to E')
plt.ylabel(r'Diff input to I')

plt.savefig(path_plot+'crosscovarianceratio.pdf')

plt.show()

print(np.nanmin(crosscovariance_ratio.T-1))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
