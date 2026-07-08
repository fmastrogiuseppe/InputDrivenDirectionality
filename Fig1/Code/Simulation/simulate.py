
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

N = 2
tau_m = 3

# Simulations

Ntrials = 100
T = 100
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 20
bin_size = 4


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

doCompute = 0

if doCompute:

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SETUP NETWORK

	# Connectivity

	W, lambdas, P = var.RandomConnectivity(N, rescale_to = 0.4)
	Pinv = np.linalg.inv(P)
	print(lambdas)

	# Three possible types of input matrices

	U_a = np.random.normal(0, 1, (N,N)) 								# predicted: asymmetry
	U_b = P 															# predicted: symmetry
	u = np.array([2,0])
	U_c = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])		# predicted: asymmetry with delay


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### THEORY

	tau_values, cross_a_bar, cross_a, cross_a_bar_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_a, tau_m = tau_m)
	tau_values, cross_b_bar, cross_b, cross_a_bar_input, cross_a_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_b, tau_m = tau_m)
	tau_values, cross_c_bar, cross_c, cross_c_bar_input, cross_c_input = \
			th.ComputeCovarianceTheory(lambdas, P, Pinv, U_c, tau_m = tau_m)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	#

	X = var.SimulateActivity(t, X0, W, U_a)
	X = X[:,int(Tcut/deltaT):,:]

	cross_a_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X.shape[2]):
		for ii_neuron_2 in range(X.shape[2]):
			cross_a_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], X[:,:,ii_neuron_2], int(tau_m/deltaT))
	
	tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_a_sim[0,1]))

	#

	X = var.SimulateActivity(t, X0, W, U_b)
	X = X[:,int(Tcut/deltaT):,:]

	cross_b_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X.shape[2]):
		for ii_neuron_2 in range(X.shape[2]):
			cross_b_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], X[:,:,ii_neuron_2], int(tau_m/deltaT))

	#

	X = var.SimulateActivity(t, X0, W, U_c)
	X = X[:,int(Tcut/deltaT):,:]

	cross_c_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X.shape[2]):
		for ii_neuron_2 in range(X.shape[2]):
			cross_c_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], X[:,:,ii_neuron_2], int(tau_m/deltaT))

	# Some sample binned traces, for visualization purposes 

	X_binned = var.Bin(X, bin_size)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross_a, 'cross_a.p', path_data)
	fac.Store(cross_b, 'cross_b.p', path_data)
	fac.Store(cross_c, 'cross_c.p', path_data)

	fac.Store(cross_a_sim, 'cross_a_sim.p', path_data)
	fac.Store(cross_b_sim, 'cross_b_sim.p', path_data)
	fac.Store(cross_c_sim, 'cross_c_sim.p', path_data)

	fac.Store(X_binned, 'X_binned.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross_a = fac.Retrieve('cross_a.p', path_data)
	cross_b = fac.Retrieve('cross_b.p', path_data)
	cross_c = fac.Retrieve('cross_c.p', path_data)

	cross_a_sim = fac.Retrieve('cross_a_sim.p', path_data)
	cross_b_sim = fac.Retrieve('cross_b_sim.p', path_data)
	cross_c_sim = fac.Retrieve('cross_c_sim.p', path_data)

	X_binned = fac.Retrieve('X_binned.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

#

Ncolumns = 3
fac.SetPlotDim(1.9*Ncolumns, 1.9)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.subplot(1, Ncolumns, 1)
plt.plot(tau_values, cross_a[0,1], color = '0')
plt.plot(tau_values, np.flipud(cross_a[0,1]), color = '0', ls='--', linewidth=0.5)
plt.plot(tau_values_sim, cross_a_sim[0,1], color='r', linewidth=0.5, label='sim')
plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')
plt.title('Random $U$')
plt.legend(loc=4, frameon=False)

plt.subplot(1, Ncolumns, 2)
plt.plot(tau_values, cross_b[0,1], color = '0')
plt.plot(tau_values, np.flipud(cross_b[0,1]), color = '0', ls='--', linewidth=0.5)
plt.plot(tau_values_sim, cross_b_sim[0,1], color='r', linewidth=0.5)
plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')
plt.title('$U=P$')

plt.subplot(1, Ncolumns, 3)
plt.plot(tau_values, cross_c[0,1], color = '0')
plt.plot(tau_values, np.flipud(cross_c[0,1]), color = '0', ls='--', linewidth=0.5)
plt.plot(tau_values_sim, cross_c_sim[0,1], color='r', linewidth=0.5)
plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')
plt.title('$U=$low-d')

plt.savefig(path_plot+'cross_all.pdf')

plt.show()

#

fac.SetPlotDim(1.8, 1.65)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)
plt.axvline(x=tau_values_sim[np.argmax(cross_c_sim[0,1])], color='#FF1A66', ls='--')

plt.fill_between(tau_values_sim[tau_values_sim<=0], (cross_c_sim[0,1])[tau_values_sim<=0], alpha=0.3, color='#E50058')
plt.fill_between(tau_values_sim[tau_values_sim>=0], (cross_c_sim[0,1])[tau_values_sim>=0], alpha=0.3, color='#FF9999')

plt.plot(tau_values_sim, cross_c_sim[0,1], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 0.4)
plt.yticks([0, 0.2, 0.4], ['$0.0$', '$0.5$', '$1.0$'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig(path_plot+'cross.pdf')

plt.show()

#

Tlim = 15

fac.SetPlotDim(1.85, 1.)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_binned[0,:int(Tlim/deltaT),0], color='#00A89F')

plt.xlabel(r'Time')
plt.ylabel(r'Activity')

plt.xlim(0, Tlim)
plt.ylim(-3, 3)
plt.yticks([-3, 3], ['$0$', '$1$'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig(path_plot+'unit1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_binned[0,:int(Tlim/deltaT),1], color='#FF7F00')

plt.xlabel(r'Time')
plt.ylabel(r'Activity')

plt.xlim(0, Tlim)
plt.ylim(-0.7, 0.7)
plt.yticks([-0.7, 0.7], ['$0$', '$1$'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig(path_plot+'unit2.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
