
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

N = 4
tau_m = 3

# Simulations

Ntrials = 100
T = 420
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 20


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

doCompute = 1

if doCompute:

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SETUP NETWORK

	# Connectivity

	W, lambdas, P = var.RandomConnectivity(N, rescale_to=0.58)
	Pinv = np.linalg.inv(P)
	print(lambdas)

	# Possible types of input matrices

	d = np.random.uniform(0, 5, (N))
	R = var.OrthogonalBasis(N,N)

	U_a = np.random.normal(0, 1, (N,N))					# predicted: asymmetry
	U_b = P												# predicted: symmetry
	U_c = np.dot(P, np.diag(np.sqrt(d)))				# predicted: symmetry
	U_d = np.dot( np.dot(P, np.diag(np.sqrt(d))), R )	# predicted: symmetry


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### THEORY

	tau_values, crossbar_a, cross_a, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_a, tau_m = tau_m)
	tau_values, crossbar_b, cross_b, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_b, tau_m = tau_m)
	tau_values, crossbar_c, cross_c, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_c, tau_m = tau_m)
	tau_values, crossbar_d, cross_d, crossbar_a_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_d, tau_m = tau_m)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	#

	X_a, Y_a, eta = var.SimulateActivity(t, X0, W, U_a, return_noise=True)
	X_a, Y_a = X_a[:,int(Tcut/deltaT):,:], Y_a[:,int(Tcut/deltaT):,:]

	cross_a_sim = np.zeros(( X_a.shape[2], X_a.shape[2], len(var.ComputeCovariance(X_a[:,:,0], X_a[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_a.shape[2]):
		for ii_neuron_2 in range(X_a.shape[2]):

			cross_a_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_a[:,:,ii_neuron_1], X_a[:,:,ii_neuron_2], int(tau_m/deltaT))

			if ii_neuron_1 == ii_neuron_2 == 0: tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_a_sim[0,1]))

			asymmetry_a = th.AsymmetryScore(cross_a_sim, tau_values_sim)[~np.eye(N).astype(bool)]
			lag_a = th.PrincipalLag(cross_a_sim, tau_values_sim)[~np.eye(N).astype(bool)]

	#

	X_b, Y_b, eta = var.SimulateActivity(t, X0, W, U_b, return_noise=True)
	X_b, Y_b = X_b[:,int(Tcut/deltaT):,:], Y_b[:,int(Tcut/deltaT):,:]

	cross_b_sim = np.zeros(( X_b.shape[2], X_b.shape[2], len(var.ComputeCovariance(X_b[:,:,0], X_b[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_b.shape[2]):
		for ii_neuron_2 in range(X_b.shape[2]):
			
			cross_b_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_b[:,:,ii_neuron_1], X_b[:,:,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_b = th.AsymmetryScore(cross_b_sim, tau_values_sim)[~np.eye(N).astype(bool)]
			lag_b = th.PrincipalLag(cross_b_sim, tau_values_sim)[~np.eye(N).astype(bool)]

	#

	X_c, Y_c, eta = var.SimulateActivity(t, X0, W, U_c, return_noise=True)
	X_c, Y_c = X_c[:,int(Tcut/deltaT):,:], Y_c[:,int(Tcut/deltaT):,:]

	cross_c_sim = np.zeros(( X_c.shape[2], X_c.shape[2], len(var.ComputeCovariance(X_c[:,:,0], X_c[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_c.shape[2]):
		for ii_neuron_2 in range(X_c.shape[2]):
			
			cross_c_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_c[:,:,ii_neuron_1], X_c[:,:,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_c = th.AsymmetryScore(cross_c_sim, tau_values_sim)[~np.eye(N).astype(bool)]
			lag_c = th.PrincipalLag(cross_c_sim, tau_values_sim)[~np.eye(N).astype(bool)]


	#

	X_d, Y_d, eta = var.SimulateActivity(t, X0, W, U_d, return_noise=True)
	X_d, Y_d = X_d[:,int(Tcut/deltaT):,:], Y_d[:,int(Tcut/deltaT):,:]

	cross_d_sim = np.zeros(( X_d.shape[2], X_d.shape[2], len(var.ComputeCovariance(X_d[:,:,0], X_d[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X_d.shape[2]):
		for ii_neuron_2 in range(X_d.shape[2]):
			
			cross_d_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_d[:,:,ii_neuron_1], X_d[:,:,ii_neuron_2], int(tau_m/deltaT))
	
			asymmetry_d = th.AsymmetryScore(cross_d_sim, tau_values_sim)[~np.eye(N).astype(bool)]
			lag_d = th.PrincipalLag(cross_d_sim, tau_values_sim)[~np.eye(N).astype(bool)]


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross_a, 'cross_a.p', path_data)
	fac.Store(cross_b, 'cross_b.p', path_data)
	fac.Store(cross_c, 'cross_c.p', path_data)
	fac.Store(cross_d, 'cross_d.p', path_data)

	fac.Store(cross_a_sim, 'cross_a_sim.p', path_data)
	fac.Store(cross_b_sim, 'cross_b_sim.p', path_data)
	fac.Store(cross_c_sim, 'cross_c_sim.p', path_data)
	fac.Store(cross_d_sim, 'cross_d_sim.p', path_data)

	fac.Store(asymmetry_a, 'asymmetry_a.p', path_data)
	fac.Store(lag_a, 'lag_a.p', path_data)
	fac.Store(asymmetry_b, 'asymmetry_b.p', path_data)
	fac.Store(lag_b, 'lag_b.p', path_data)
	fac.Store(asymmetry_c, 'asymmetry_c.p', path_data)
	fac.Store(lag_c, 'lag_c.p', path_data)
	fac.Store(asymmetry_d, 'asymmetry_d.p', path_data)
	fac.Store(lag_d, 'lag_d.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross_a = fac.Retrieve('cross_a.p', path_data)
	cross_b = fac.Retrieve('cross_b.p', path_data)
	cross_c = fac.Retrieve('cross_c.p', path_data)
	cross_d = fac.Retrieve('cross_d.p', path_data)

	cross_a_sim = fac.Retrieve('cross_a_sim.p', path_data)
	cross_b_sim = fac.Retrieve('cross_b_sim.p', path_data)
	cross_c_sim = fac.Retrieve('cross_c_sim.p', path_data)
	cross_d_sim = fac.Retrieve('cross_d_sim.p', path_data)

	asymmetry_a = fac.Retrieve('asymmetry_a.p', path_data)
	lag_a = fac.Retrieve('lag_a.p', path_data)
	asymmetry_b = fac.Retrieve('asymmetry_b.p', path_data)
	lag_b = fac.Retrieve('lag_b.p', path_data)
	asymmetry_c = fac.Retrieve('asymmetry_c.p', path_data)
	lag_c = fac.Retrieve('lag_c.p', path_data)
	asymmetry_d = fac.Retrieve('asymmetry_d.p', path_data)
	lag_d = fac.Retrieve('lag_d.p', path_data)


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

#

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values, cross_c[ii_neuron_1,ii_neuron_2], color = '0')
		plt.plot(tau_values, np.flipud(cross_c[ii_neuron_1,ii_neuron_2]), color = '0', ls='--', linewidth=0.5)
		plt.plot(tau_values_sim, cross_c_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')
		plt.plot(tau_values_sim, np.flipud(cross_c_sim[ii_neuron_1,ii_neuron_2,:]), ls='--', color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')

plt.savefig(path_plot+'crossall_c.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values, cross_d[ii_neuron_1,ii_neuron_2], color = '0')
		plt.plot(tau_values, np.flipud(cross_d[ii_neuron_1,ii_neuron_2]), color = '0', ls='--', linewidth=0.5)
		plt.plot(tau_values_sim, cross_d_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')
		plt.plot(tau_values_sim, np.flipud(cross_d_sim[ii_neuron_1,ii_neuron_2,:]), ls='--', color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')

plt.savefig(path_plot+'crossall_d.pdf')

plt.show()

#

Nrows = N
Ncolumns = N
fac.SetPlotDim(1.65*Ncolumns, 1.5*Nrows)

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values_sim, cross_a_sim[ii_neuron_1,ii_neuron_2,:], color='0.7', linewidth = 4)

		plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

		plt.plot(tau_values, cross_a[ii_neuron_1,ii_neuron_2], color = '0.2')

		plt.xlim(-tau_m, tau_m)
		plt.xticks([-tau_m, 0, tau_m])

		# plt.ylim(0, 1.8)
		# plt.yticks([0, 0.9, 1.8])

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')

		plt.locator_params(nbins=3)

plt.savefig(path_plot+'crossall_a_polished.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Cross-covariances

ii_neuron_1 = 0
ii_neuron_2 = 3

#

fac.SetPlotDim(1.6, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, cross_a_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_a[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 4.4)
plt.yticks([0, 2.2, 4.4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, cross_b_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_b[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 0.4)
plt.yticks([0, 0.2, 0.4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, cross_c_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_c[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 0.6)
plt.yticks([0, 0.3, 0.6])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_c.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, cross_d_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_d[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 0.4)
plt.yticks([0, 0.2, 0.4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_d.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Asymmetry measures

fac.SetPlotDim(1.65, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='0', linewidth=0.7)

plt.plot(np.zeros(len(asymmetry_a)), asymmetry_a, 'o', color = '0.4', markersize = 4)
plt.plot(np.ones(len(asymmetry_b)), asymmetry_b, 'o', color = '0.4', markersize = 4)
plt.plot(1+np.ones(len(asymmetry_c)), asymmetry_c, 'o', color = '0.4', markersize = 4)
plt.plot(2+np.ones(len(asymmetry_d)), asymmetry_d, 'o', color = '0.4', markersize = 4)

plt.plot(np.zeros(len(asymmetry_a)), asymmetry_a, 'o', color = '0.7', markersize = 2)
plt.plot(np.ones(len(asymmetry_b)), asymmetry_b, 'o', color = '0.7', markersize = 2)
plt.plot(1+np.ones(len(asymmetry_c)), asymmetry_c, 'o', color = '0.7', markersize = 2)
plt.plot(2+np.ones(len(asymmetry_d)), asymmetry_d, 'o', color = '0.7', markersize = 2)

plt.xlabel(r'Input configuration')
plt.ylabel(r'Asymmetry')

plt.xlim(-0.9, 3.9)
plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'])

limit = 3
plt.ylim(-limit, limit)
plt.yticks([-limit, 0, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig(path_plot+'asymmetry.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
