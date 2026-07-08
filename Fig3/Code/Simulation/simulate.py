
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
T = 80
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

	W, lambdas, P = var.RandomConnectivity(N, lambda_low=-0.1, lambda_high=0.5, rescale_to=0.58)
	Pinv = np.linalg.inv(P)
	print(lambdas)

	# Two possible types of input matrices

	u = P[:,0]+P[:,2]
	U_a = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])		# predicted: asymmetry with delay

	u = P[:,0]
	U_b = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])		# predicted: symmetry


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
	X_a, Y_a = X_a[:,int(Tcut/deltaT):,:], Y_a[:,int(Tcut/deltaT):,:]

	Xbar_a = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(X_a, 0, 2), (X_a.shape[2], X_a.shape[0]*X_a.shape[1])) ), \
		(X_a.shape[2], X_a.shape[1], X_a.shape[0]) ), 0, 2)
	Ybar_a = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(Y_a, 0, 2), (X_a.shape[2], X_a.shape[0]*X_a.shape[1])) ), \
		(X_a.shape[2], X_a.shape[1], X_a.shape[0]) ), 0, 2)

	cross_a_sim = np.zeros(( X_a.shape[2], X_a.shape[2], len(var.ComputeCovariance(X_a[:,:,0], X_a[:,:,0], int(tau_m/deltaT))) ))
	crossbar_a_sim = np.zeros_like(cross_a_sim)

	for ii_neuron_1 in range(X_a.shape[2]):
		for ii_neuron_2 in range(X_a.shape[2]):
			cross_a_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_a[:,:,ii_neuron_1], X_a[:,:,ii_neuron_2], int(tau_m/deltaT))
			crossbar_a_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(Xbar_a[:,:,ii_neuron_1], Xbar_a[:,:,ii_neuron_2], int(tau_m/deltaT))

	tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_a_sim[0,1]))

	#

	X_b, Y_b, eta = var.SimulateActivity(t, X0, W, U_b, eta, target_noise=True, return_noise=True)
	X_b, Y_b = X_b[:,int(Tcut/deltaT):,:], Y_b[:,int(Tcut/deltaT):,:]
	
	Xbar_b = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(X_b, 0, 2), (X_b.shape[2], X_b.shape[0]*X_b.shape[1])) ), \
		(X_b.shape[2], X_b.shape[1], X_b.shape[0]) ), 0, 2)
	Ybar_b = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(Y_b, 0, 2), (X_b.shape[2], X_b.shape[0]*X_b.shape[1])) ), \
		(X_b.shape[2], X_b.shape[1], X_b.shape[0]) ), 0, 2)

	cross_b_sim = np.zeros(( X_b.shape[2], X_b.shape[2], len(var.ComputeCovariance(X_b[:,:,0], X_b[:,:,0], int(tau_m/deltaT))) ))
	crossbar_b_sim = np.zeros_like(cross_b_sim)

	for ii_neuron_1 in range(X_b.shape[2]):
		for ii_neuron_2 in range(X_b.shape[2]):
			cross_b_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X_b[:,:,ii_neuron_1], X_b[:,:,ii_neuron_2], int(tau_m/deltaT))
			crossbar_b_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(Xbar_b[:,:,ii_neuron_1], Xbar_b[:,:,ii_neuron_2], int(tau_m/deltaT))
			

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross_a, 'cross_a.p', path_data)
	fac.Store(cross_b, 'cross_b.p', path_data)
	fac.Store(crossbar_a, 'crossbar_a.p', path_data)
	fac.Store(crossbar_b, 'crossbar_b.p', path_data)

	fac.Store(cross_a_sim, 'cross_a_sim.p', path_data)
	fac.Store(cross_b_sim, 'cross_b_sim.p', path_data)
	fac.Store(crossbar_a_sim, 'crossbar_a_sim.p', path_data)
	fac.Store(crossbar_b_sim, 'crossbar_b_sim.p', path_data)

	fac.Store(X_a, 'X_a.p', path_data)
	fac.Store(Y_a, 'Y_a.p', path_data)
	fac.Store(X_b, 'X_b.p', path_data)
	fac.Store(Y_b, 'Y_b.p', path_data)

	fac.Store(Xbar_a, 'Xbar_a.p', path_data)
	fac.Store(Ybar_a, 'Ybar_a.p', path_data)
	fac.Store(Xbar_b, 'Xbar_b.p', path_data)
	fac.Store(Ybar_b, 'Ybar_b.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross_a = fac.Retrieve('cross_a.p', path_data)
	cross_b = fac.Retrieve('cross_b.p', path_data)
	crossbar_a = fac.Retrieve('crossbar_a.p', path_data)
	crossbar_b = fac.Retrieve('crossbar_b.p', path_data)

	cross_a_sim = fac.Retrieve('cross_a_sim.p', path_data)
	cross_b_sim = fac.Retrieve('cross_b_sim.p', path_data)
	crossbar_a_sim = fac.Retrieve('crossbar_a_sim.p', path_data)
	crossbar_b_sim = fac.Retrieve('crossbar_b_sim.p', path_data)

	X_a = fac.Retrieve('X_a.p', path_data)
	Y_a = fac.Retrieve('Y_a.p', path_data)
	X_b = fac.Retrieve('X_b.p', path_data)
	Y_b = fac.Retrieve('Y_b.p', path_data)

	Xbar_a = fac.Retrieve('Xbar_a.p', path_data)
	Ybar_a = fac.Retrieve('Ybar_a.p', path_data)
	Xbar_b = fac.Retrieve('Xbar_b.p', path_data)
	Ybar_b = fac.Retrieve('Ybar_b.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

Tlim = 15

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
ii_neuron_2 = 2

#

fac.SetPlotDim(1.6, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, cross_b_sim[ii_neuron_1,ii_neuron_2], color='#E5CCFF', linewidth = 2)
plt.plot(tau_values_sim, cross_a_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_b[ii_neuron_1,ii_neuron_2], color='#BC68EC')
plt.plot(tau_values, cross_a[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 0.7)
plt.yticks([0, 0.7])

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

plt.plot(tau_values_sim, cross_b_sim[ii_neuron_1,ii_neuron_2], color='#E5CCFF', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, cross_b[ii_neuron_1,ii_neuron_2], color='#BC68EC')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

# plt.ylim(0, 0.05)
# plt.yticks([0, 0.05])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_b.pdf')

plt.show()

#

fac.SetPlotDim(1.45, 0.95)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_a[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, Tlim)

limit = 2.6
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X1_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, Tlim)

limit = 1.4
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X2_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Y_a[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, Tlim)

limit = 4.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y1_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Y_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, Tlim)

limit = 0.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y2_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_b[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, Tlim)

limit = 0.3
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X1_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X_b[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, Tlim)

limit = 1.9
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X2_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Y_b[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 1')

plt.xlim(0, Tlim)

limit = 0.3
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y1_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Y_b[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Unit 2')

plt.xlim(0, Tlim)

limit = 2.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y2_b.pdf')

plt.show()

#

fac.SetPlotDim(0.65, 0.65)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='k', linewidth=0.7)
plt.axvline(x=0, color='k', linewidth=0.7)

plt.plot(Y_a[0,:int(Tlim/deltaT),ii_neuron_1], Y_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=1)
plt.plot(Y_b[0,:int(Tlim/deltaT),ii_neuron_1], Y_b[0,:int(Tlim/deltaT),ii_neuron_2], color='#BC68EC', linewidth=1)

lim = 3.8
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.axis('off')

plt.savefig(path_plot+'Y1Y2.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Eigenvector space

ii_neuron_1 = 0
ii_neuron_2 = 2

#

fac.SetPlotDim(1.6, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, crossbar_b_sim[ii_neuron_1,ii_neuron_2], color='#E5CCFF', linewidth = 2)
plt.plot(tau_values_sim, crossbar_a_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, crossbar_b[ii_neuron_1,ii_neuron_2], color='#BC68EC')
plt.plot(tau_values, crossbar_a[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(-0.02, 0.7)
plt.yticks([0, 0.7])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'crossbar_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(tau_values_sim, crossbar_b_sim[ii_neuron_1,ii_neuron_2], color='#E5CCFF', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, crossbar_b[ii_neuron_1,ii_neuron_2], color='#BC68EC')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(-0.003, 0.08)
plt.yticks([0, 0.04, 0.08])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'crossbar_b.pdf')

plt.show()

#

fac.SetPlotDim(1.45, 0.95)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xbar_a[0,:int(Tlim/deltaT),ii_neuron_1], color='0.4', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 1')

plt.xlim(0, Tlim)

limit = 2.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xbar1_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xbar_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.', linewidth=0.6)
plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xbar_b[0,:int(Tlim/deltaT),ii_neuron_2], color='#BC68EC', linewidth=1)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 2')

plt.xlim(0, Tlim)

limit = 1.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xbar2_a.pdf')

plt.show()

#

ii_neuron_1 = 0
ii_neuron_2 = 2

fac.SetPlotDim(2.2, 0.7)

fg = plt.figure()
ax = plt.axes(frameon=True)

signal1 = Xbar_a[0,0:int(Tlim/deltaT),ii_neuron_1]
signal2 = Xbar_a[0,0:int(Tlim/deltaT),ii_neuron_2]

signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), signal1, color='0.4', linewidth=0.7)
plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), signal2/np.min(signal2)*np.min(signal1), color='0.', linewidth=0.7)

plt.axis('off')

plt.savefig(path_plot+'Xbar_comparison_a.pdf')

plt.show()

#

fac.SetPlotDim(1.45, 0.95)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ybar_a[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 1')

plt.xlim(0, Tlim)

limit = 3.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Ybar1_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ybar_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)
plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ybar_b[0,:int(Tlim/deltaT),ii_neuron_2], color='#BC68EC', linewidth=1)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 2')

plt.xlim(0, Tlim)

limit = 3.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Ybar2_a.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xbar_b[0,:int(Tlim/deltaT),ii_neuron_1], color='0.4', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 1')

plt.xlim(0, Tlim)

limit = 3.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xbar1_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xbar_b[0,:int(Tlim/deltaT),ii_neuron_2], color='0.', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 2')

plt.xlim(0, Tlim)

limit = 0.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xbar2_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ybar_b[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 1')

plt.xlim(0, Tlim)

limit = 3.4
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Ybar1_b.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ybar_b[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Eigen 2')

plt.xlim(0, Tlim)

limit = 0.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Ybar2_b.pdf')

plt.show()

#

fac.SetPlotDim(0.65, 0.65)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, color='k', linewidth=0.7)
plt.axvline(x=0, color='k', linewidth=0.7)

plt.plot(Ybar_a[0,:int(Tlim/deltaT),ii_neuron_1], Ybar_a[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=1)
plt.plot(Ybar_b[0,:int(Tlim/deltaT),ii_neuron_1], Ybar_b[0,:int(Tlim/deltaT),ii_neuron_2], color='#BC68EC', linewidth=1)

lim = 4.4
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.axis('off')

plt.savefig(path_plot+'Ybar1Ybar2.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
