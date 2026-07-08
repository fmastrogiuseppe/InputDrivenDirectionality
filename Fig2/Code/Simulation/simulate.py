
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

path_data = 'Data_simulate/'
path_plot = 'Plots_simulate/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 8
tau_m = 3

# Simulations

Ntrials = 100
T = 30
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 10



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

doCompute = 1

area_mask = np.zeros(( N, N )).astype(bool)
area_mask[:int(N/2), int(N/2):] = True

if doCompute:

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SETUP NETWORK

	# Connectivity

	W, lambdas, P = var.RandomConnectivity(N, rescale_to=0.58)
	Pinv = np.linalg.inv(P)
	print(lambdas)

	# Two possible types of input matrices

	u_a = np.random.normal(0, 1, N)
	U_a = np.hstack([ np.reshape(u_a, (N,1)), np.zeros((N, N-1)) ])		# predicted: asymmetry with delay

	u_b = P[:,0]
	U_b = np.hstack([ np.reshape(u_b, (N,1)), np.zeros((N, N-1)) ])		# predicted: symmetry


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### THEORY

	tau_values, crossbar_a, cross_a, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_a, tau_m = tau_m)
	tau_values, crossbar_b, cross_b, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_b, tau_m = tau_m)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	#

	X_a, Y_a, eta = var.SimulateActivity(t, X0, W, U_a, return_noise=True)

	cross_a_sim = np.zeros(( X_a.shape[2], X_a.shape[2], len(var.ComputeCovariance(X_a[:,int(Tcut/deltaT):,0], X_a[:,int(Tcut/deltaT):,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_a.shape[2]):
		for ii_neuron_2 in range(X_a.shape[2]):

			cross_a_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_a[:,int(Tcut/deltaT):,ii_neuron_1], X_a[:,int(Tcut/deltaT):,ii_neuron_2], int(tau_m/deltaT))

			if ii_neuron_1 == ii_neuron_2 == 0: tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_a_sim[0,1]))

			asymmetry_a = th.AsymmetryScore(cross_a_sim, tau_values_sim)[area_mask]
			lag_a = th.PrincipalLag(cross_a_sim, tau_values_sim)[area_mask]

	#

	X_b, Y_b, eta = var.SimulateActivity(t, X_a[:,-1,:], W, U_b, return_noise=True)

	cross_b_sim = np.zeros(( X_b.shape[2], X_b.shape[2], len(var.ComputeCovariance(X_b[:,int(Tcut/deltaT):,0], X_b[:,int(Tcut/deltaT):,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_b.shape[2]):
		for ii_neuron_2 in range(X_b.shape[2]):
			
			cross_b_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_b[:,int(Tcut/deltaT):,ii_neuron_1], X_b[:,int(Tcut/deltaT):,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_b = th.AsymmetryScore(cross_b_sim, tau_values_sim)[area_mask]
			lag_b = th.PrincipalLag(cross_b_sim, tau_values_sim)[area_mask]

	#

	X = np.concatenate((X_a, X_b), axis=1)
	Y = np.concatenate((Y_a, Y_b), axis=1)

	ubar_a = np.dot(Pinv, u_a)
	ubar_b = np.dot(Pinv, u_b)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SHUFFLE NETWORK

	# Connectivity

	W, lambdas, P = var.ShuffleConnectivity(W, rescale_to=0.58)
	Pinv = np.linalg.inv(P)
	print(lambdas)

	# Three possible types of input matrices

	u_c = P[:,0]
	U_c = np.hstack([ np.reshape(u_c, (N,1)), np.zeros((N, N-1)) ])		# predicted: symmetry


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	#

	X_a, Y_a, eta = var.SimulateActivity(t, X0, W, U_a, return_noise=True)

	cross_sim = np.zeros(( X_a.shape[2], X_a.shape[2], len(var.ComputeCovariance(X_a[:,int(Tcut/deltaT):,0], X_a[:,int(Tcut/deltaT):,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_a.shape[2]):
		for ii_neuron_2 in range(X_a.shape[2]):

			cross_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_a[:,int(Tcut/deltaT):,ii_neuron_1], X_a[:,int(Tcut/deltaT):,ii_neuron_2], int(tau_m/deltaT))

			if ii_neuron_1 == ii_neuron_2 == 0: tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_sim[0,1]))

			asymmetry_shuffle_a = th.AsymmetryScore(cross_sim, tau_values_sim)[area_mask]
			lag_shuffle_a = th.PrincipalLag(cross_sim, tau_values_sim)[area_mask]

	#

	X_b, Y_b, eta = var.SimulateActivity(t, X_a[-1,:], W, U_b, return_noise=True)

	cross_sim = np.zeros(( X_b.shape[2], X_b.shape[2], len(var.ComputeCovariance(X_b[:,:,0], X_b[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_b.shape[2]):
		for ii_neuron_2 in range(X_b.shape[2]):
			
			cross_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_b[:,int(Tcut/deltaT):,ii_neuron_1], X_b[:,int(Tcut/deltaT):,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_shuffle_b = th.AsymmetryScore(cross_sim, tau_values_sim)[area_mask]
			lag_shuffle_b = th.PrincipalLag(cross_sim, tau_values_sim)[area_mask]

	#

	X_c, Y_c, eta = var.SimulateActivity(t, X_b[-1,:], W, U_c, return_noise=True)

	cross_sim = np.zeros(( X_c.shape[2], X_c.shape[2], len(var.ComputeCovariance(X_c[:,:,0], X_c[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_c.shape[2]):
		for ii_neuron_2 in range(X_c.shape[2]):
			
			cross_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_c[:,int(Tcut/deltaT):,ii_neuron_1], X_c[:,int(Tcut/deltaT):,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_shuffle_c = th.AsymmetryScore(cross_sim, tau_values_sim)[area_mask]
			lag_shuffle_c = th.PrincipalLag(cross_sim, tau_values_sim)[area_mask]

	#

	ubar_shuffle_a = np.dot(Pinv, u_a)
	ubar_shuffle_b = np.dot(Pinv, u_b)
	ubar_shuffle_c = np.dot(Pinv, u_c)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross_a, 'cross_a.p', path_data)
	fac.Store(cross_b, 'cross_b.p', path_data)
	fac.Store(cross_a_sim, 'cross_a_sim.p', path_data)
	fac.Store(cross_b_sim, 'cross_b_sim.p', path_data)

	fac.Store(asymmetry_a, 'asymmetry_a.p', path_data)
	fac.Store(lag_a, 'lag_a.p', path_data)
	fac.Store(asymmetry_b, 'asymmetry_b.p', path_data)
	fac.Store(lag_b, 'lag_b.p', path_data)

	fac.Store(X, 'X.p', path_data)
	fac.Store(Y, 'Y.p', path_data)

	fac.Store(ubar_a, 'ubar_a.p', path_data)
	fac.Store(ubar_b, 'ubar_b.p', path_data)

	fac.Store(asymmetry_shuffle_a, 'asymmetry_shuffle_a.p', path_data)
	fac.Store(lag_shuffle_a, 'lag_shuffle_a.p', path_data)
	fac.Store(asymmetry_shuffle_b, 'asymmetry_shuffle_b.p', path_data)
	fac.Store(lag_shuffle_b, 'lag_shuffle_b.p', path_data)
	fac.Store(asymmetry_shuffle_c, 'asymmetry_shuffle_c.p', path_data)
	fac.Store(lag_shuffle_c, 'lag_shuffle_c.p', path_data)

	fac.Store(ubar_shuffle_a, 'ubar_shuffle_a.p', path_data)
	fac.Store(ubar_shuffle_b, 'ubar_shuffle_b.p', path_data)
	fac.Store(ubar_shuffle_c, 'ubar_shuffle_c.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross_a = fac.Retrieve('cross_a.p', path_data)
	cross_b = fac.Retrieve('cross_b.p', path_data)
	cross_a_sim = fac.Retrieve('cross_a_sim.p', path_data)
	cross_b_sim = fac.Retrieve('cross_b_sim.p', path_data)

	asymmetry_a = fac.Retrieve('asymmetry_a.p', path_data)
	lag_a = fac.Retrieve('lag_a.p', path_data)
	asymmetry_b = fac.Retrieve('asymmetry_b.p', path_data)
	lag_b = fac.Retrieve('lag_b.p', path_data)

	X = fac.Retrieve('X.p', path_data)
	Y = fac.Retrieve('Y.p', path_data)

	ubar_a = fac.Retrieve('ubar_a.p', path_data)
	ubar_b = fac.Retrieve('ubar_b.p', path_data)

	asymmetry_shuffle_a = fac.Retrieve('asymmetry_shuffle_a.p', path_data)
	lag_shuffle_a = fac.Retrieve('lag_shuffle_a.p', path_data)
	asymmetry_shuffle_b = fac.Retrieve('asymmetry_shuffle_b.p', path_data)
	lag_shuffle_b = fac.Retrieve('lag_shuffle_b.p', path_data)
	asymmetry_shuffle_c = fac.Retrieve('asymmetry_shuffle_c.p', path_data)
	lag_shuffle_c = fac.Retrieve('lag_shuffle_c.p', path_data)

	ubar_shuffle_a = fac.Retrieve('ubar_shuffle_a.p', path_data)
	ubar_shuffle_b = fac.Retrieve('ubar_shuffle_b.p', path_data)
	ubar_shuffle_c = fac.Retrieve('ubar_shuffle_c.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()


#

Nrows = N
Ncolumns = N
fac.SetPlotDim(1.9*Ncolumns, 1.7*Nrows)

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values, cross_a[ii_neuron_1,ii_neuron_2], color = '0')
		plt.plot(tau_values, np.flipud(cross_a[ii_neuron_1,ii_neuron_2]), color = '0', ls='--', linewidth=0.5)
		plt.plot(tau_values_sim, cross_a_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')
		plt.plot(tau_values_sim, np.flipud(cross_a_sim[ii_neuron_1,ii_neuron_2,:]), ls='--', color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')

plt.savefig(path_plot+'crossall_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values, cross_b[ii_neuron_1,ii_neuron_2], color = '0')
		plt.plot(tau_values, np.flipud(cross_b[ii_neuron_1,ii_neuron_2]), color = '0', ls='--', linewidth=0.5)
		plt.plot(tau_values_sim, cross_b_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')
		plt.plot(tau_values_sim, np.flipud(cross_b_sim[ii_neuron_1,ii_neuron_2,:]), ls='--', color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')

plt.savefig(path_plot+'crossall_b.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Neural space

ii_neuron_1 = 0
ii_neuron_2 = 6

# 0-5

#

fac.SetPlotDim(1.45, 0.95)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, 2*T, int(2*T/deltaT)), X[0,:,ii_neuron_1], color='0.6', linewidth=0.6)
plt.axvline(x=T, color='0', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, 2*T)
plt.xticks([0, T, 2*T])

limit = 0.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, 2*T, int(2*T/deltaT)), X[0,:,ii_neuron_2], color='0.6', linewidth=0.6)
plt.axvline(x=T, color='0', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, 2*T)
plt.xticks([0, T, 2*T])

limit = 1.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X2.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, 2*T, int(2*T/deltaT)), Y[0,:,ii_neuron_1], color='0.6', linewidth=0.6)
plt.axvline(x=T, color='0', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, 2*T)
plt.xticks([0, T, 2*T])

limit = 0.9
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, 2*T, int(2*T/deltaT)), Y[0,:,ii_neuron_2], color='0.6', linewidth=0.6)
plt.axvline(x=T, color='0', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, 2*T)
plt.xticks([0, T, 2*T])

limit = 1.1
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y2.pdf')

plt.show()


# Asymmetry measures

ii_pair = ii_neuron_1*int(N/2) + (ii_neuron_2-int(N/2))

fac.SetPlotDim(1.35, 1.65)

fg = plt.figure()
ax = plt.axes(frameon=True)

# plt.plot(np.zeros(len(asymmetry_a)), asymmetry_a, 'o', color = '#FF1A66', markersize = 4, alpha=0.5)
# plt.plot(np.ones(len(asymmetry_b)), asymmetry_b, 'o', color = '#BC68EC', markersize = 4, alpha=0.5)

plt.plot(np.zeros(len(asymmetry_a)), asymmetry_a, 'o', color = '#FFBAD2', markersize = 3)
plt.plot(np.ones(len(asymmetry_b)), asymmetry_b, 'o', color = '#E5CCFF', markersize = 3)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(0, asymmetry_a[ii_pair], 'o', color = '#FF1A66', markersize = 5)
plt.plot(1, asymmetry_b[ii_pair], 'o', color = '#BC68EC', markersize = 5)

plt.plot(0, asymmetry_a[ii_pair], 'o', color = '#FF80AB', markersize = 3)
plt.plot(1, asymmetry_b[ii_pair], 'o', color = '#CD9AFF', markersize = 3)

plt.xlabel(r'Epoch')
plt.ylabel(r'Asymmetry score')

plt.xlim(-0.7, 1.7)
plt.xticks([0, 1], ['1', '2'])

limit = 1.5
plt.ylim(-limit, limit)
plt.yticks([-limit, 0, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'asymmetry.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='0', linewidth=0.7)

# plt.plot(np.zeros(len(asymmetry_a)), lag_a, 'o', color = '#FF1A66', markersize = 4)
# plt.plot(np.ones(len(asymmetry_b)), lag_b, 'o', color = '#BC68EC', markersize = 4)

plt.plot(np.zeros(len(asymmetry_a)), lag_a, 'o', color = '#FFBAD2', markersize = 3)
plt.plot(np.ones(len(asymmetry_b)), lag_b, 'o', color = '#E5CCFF', markersize = 3)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(0, lag_a[ii_pair], 'o', color = '#FF1A66', markersize = 5)
plt.plot(1, lag_b[ii_pair], 'o', color = '#BC68EC', markersize = 5)

plt.plot(0, lag_a[ii_pair], 'o', color = '#FF80AB', markersize = 3)
plt.plot(1, lag_b[ii_pair], 'o', color = '#CD9AFF', markersize = 3)

plt.xlabel(r'Epoch')
plt.ylabel(r'Principal Lag')

plt.xlim(-0.7, 1.7)
plt.xticks([0, 1], ['1', '2'])

limit = 2
plt.ylim(-limit, limit)
plt.yticks([-limit, 0, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'lag.pdf')

plt.show()


# Eigenvector components

fac.SetPlotDim(1.6, 1.5)

Nmax = 6

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_a))[:Nmax], 'o', color = '#FF1A66', markersize = 4)
plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_a))[:Nmax], 'o', color = '#FFBAD2', markersize = 2)

plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_b))[:Nmax], 'o', color = '#BC68EC', markersize = 4)
plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_b))[:Nmax], 'o', color = '#E5CCFF', markersize = 2)

plt.xlabel(r'Eigenvector nbr')
plt.ylabel(r'Input component')

plt.ylim(-0.3, 1.2)
plt.yticks([0, 0.6, 1.2])

plt.xlim(0.5, Nmax+0.5)
plt.xticks(np.arange(Nmax)+1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=3)

plt.savefig(path_plot+'components.pdf')

plt.show()


# Asymmetry measures, shuffled

fac.SetPlotDim(1.45, 1.65)

fg = plt.figure()
ax = plt.axes(frameon=True)

# plt.plot(np.zeros(len(asymmetry_shuffle_a)), asymmetry_shuffle_a, 'o', color = '#FF1A66', markersize = 4)
# plt.plot(np.ones(len(asymmetry_shuffle_b)), asymmetry_shuffle_b, 'o', color = '#BC68EC', markersize = 4)
# plt.plot(1+np.ones(len(asymmetry_shuffle_b)), asymmetry_shuffle_c, 'o', color = '#4A74CB', markersize = 4)

plt.plot(np.zeros(len(asymmetry_shuffle_a)), asymmetry_shuffle_a, 'o', color = '#FFBAD2', markersize = 3)
plt.plot(np.ones(len(asymmetry_shuffle_b)), asymmetry_shuffle_b, 'o', color = '#E5CCFF', markersize = 3)
plt.plot(1+np.ones(len(asymmetry_shuffle_b)), asymmetry_shuffle_c, 'o', color = '#87A9EC', markersize = 3)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(0, asymmetry_shuffle_a[ii_pair], 'o', color = '#FF1A66', markersize = 5)
plt.plot(1, asymmetry_shuffle_b[ii_pair], 'o', color = '#BC68EC', markersize = 5)
plt.plot(2, asymmetry_shuffle_c[ii_pair], 'o', color = '#4A74CB', markersize = 5)

plt.plot(0, asymmetry_shuffle_a[ii_pair], 'o', color = '#FF80AB', markersize = 3)
plt.plot(1, asymmetry_shuffle_b[ii_pair], 'o', color = '#CD9AFF', markersize = 3)
plt.plot(2, asymmetry_shuffle_c[ii_pair], 'o', color = '#709AEF', markersize = 3)

plt.xlabel(r'Epoch')
plt.ylabel(r'Asymmetry score')

plt.xlim(-0.7, 2.7)
plt.xticks([0, 1, 2], ['1', '2', '3'])

limit = 1.5
plt.ylim(-limit, limit)
plt.yticks([-limit, 0, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'asymmetry_shuffle.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

# plt.plot(np.zeros(len(asymmetry_shuffle_a)), lag_shuffle_a, 'o', color = '#FF1A66', markersize = 4)
# plt.plot(np.ones(len(asymmetry_shuffle_b)), lag_shuffle_b, 'o', color = '#BC68EC', markersize = 4)
# plt.plot(1+np.ones(len(asymmetry_shuffle_b)), lag_shuffle_c, 'o', color = '#4A74CB', markersize = 4)

plt.plot(np.zeros(len(asymmetry_shuffle_a)), lag_shuffle_a, 'o', color = '#FFBAD2', markersize = 3)
plt.plot(np.ones(len(asymmetry_shuffle_b)), lag_shuffle_b, 'o', color = '#E5CCFF', markersize = 3)
plt.plot(1+np.ones(len(asymmetry_shuffle_b)), lag_shuffle_c, 'o', color = '#87A9EC', markersize = 3)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(0, lag_shuffle_a[ii_pair], 'o', color = '#FF1A66', markersize = 5)
plt.plot(1, lag_shuffle_b[ii_pair], 'o', color = '#BC68EC', markersize = 5)
plt.plot(2, lag_shuffle_c[ii_pair], 'o', color = '#4A74CB', markersize = 5)

plt.plot(0, lag_shuffle_a[ii_pair], 'o', color = '#FF80AB', markersize = 3)
plt.plot(1, lag_shuffle_b[ii_pair], 'o', color = '#CD9AFF', markersize = 3)
plt.plot(2, lag_shuffle_c[ii_pair], 'o', color = '#709AEF', markersize = 3)

plt.xlabel(r'Epoch')
plt.ylabel(r'Principal Lag')

plt.xlim(-0.7, 2.7)
plt.xticks([0, 1, 2], ['1', '2', '3'])

limit = 3
plt.ylim(-limit, limit)
plt.yticks([-limit, 0, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'lag_shuffle.pdf')

plt.show()


# Eigenvector components

fac.SetPlotDim(1.6, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_a))[:Nmax], 'o', color = '#FF1A66', markersize = 4)
plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_a))[:Nmax], 'o', color = '#FFBAD2', markersize = 2)

plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_b))[:Nmax], 'o', color = '#BC68EC', markersize = 4)
plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_b))[:Nmax], 'o', color = '#E5CCFF', markersize = 2)

plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_c))[:Nmax], 'o', color = '#4A74CB', markersize = 4)
plt.plot(np.arange(Nmax)+1, var.Normalize(np.fabs(ubar_shuffle_c))[:Nmax], 'o', color = '#87A9EC', markersize = 2)

plt.xlabel(r'Eigenvector nbr')
plt.ylabel(r'Input component')

plt.ylim(-0.3, 1.2)
plt.yticks([0, 0.6, 1.2])

plt.xlim(0.5, Nmax+0.5)
plt.xticks(np.arange(Nmax)+1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=3)

plt.savefig(path_plot+'components_shuffle.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
