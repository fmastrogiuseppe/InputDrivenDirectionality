
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
tau_m = 7

k = 1.01
w = 1.5
h = 0.5


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

	W = np.array( [ [ w, -k*w, h, 0 ],\
					[ w, -k*w, h, 0 ],\
					[ h, 0, w, -k*w ],\
					[ h, 0, w, -k*w ] ] )

	lambdas = np.array([ w*(1-k)+h, w*(1-k)-h, 0, 0 ])

	N3 = np.sqrt(2*(k**2+(1+h/w)**2))/k
	N4 = np.sqrt(2*(k**2+(1-h/w)**2))/k

	p1 = np.array([ 1., 1.,  1., 1.])/2
	p2 = np.array([ 1., 1.,  -1., -1.])/2
	p3 = np.array([ 1, 1./k*(1+h/w), 1, 1./k*(1+h/w)  ]) / N3
	p4 = np.array([ 1, 1./k*(1-h/w), -1, -1./k*(1-h/w)  ]) / N4
	P = (np.array([ p1, p2, p3, p4 ])).T

	Pinv = np.linalg.inv(P)
	print(lambdas)


	# Inputs

	u = np.array([ 1, 1.5, 1, 0.5 ])
	U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### THEORY

	tau_values, cross_bar, cross, crossinput_bar, crossinput = \
		th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	X, Y, eta = var.SimulateActivity(t, X0, W, U, return_noise=1)

	X = X[:,int(Tcut/deltaT):,:]
	eta = eta[:,int(Tcut/deltaT):,:]

	#

	Xbar = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(X, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(X.shape[2], X.shape[1], X.shape[0]) ), 0, 2)
	Xsignals = np.concatenate( ( (Xbar[:,:,0]/2. + Xbar[:,:,2]/N3)[:,:,None], (Xbar[:,:,1]/2. + Xbar[:,:,3]/N4)[:,:,None] ), axis=2 )

	#

	cross_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))
	crossinput_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X.shape[2]):
		for ii_neuron_2 in range(X.shape[2]):

			cross_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], X[:,:,ii_neuron_2], int(tau_m/deltaT))
			crossinput_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], eta[:,:,ii_neuron_2], int(tau_m/deltaT))

	tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_sim[0,1]))


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross, 'cross.p', path_data)
	fac.Store(cross_sim, 'cross_sim.p', path_data)
	fac.Store(crossinput, 'crossinput.p', path_data)
	fac.Store(crossinput_sim, 'crossinput_sim.p', path_data)

	fac.Store(X, 'X.p', path_data)
	fac.Store(Xbar, 'Xbar.p', path_data)
	fac.Store(Xsignals, 'Xsignals.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross = fac.Retrieve('cross.p', path_data)
	cross_sim = fac.Retrieve('cross_sim.p', path_data)
	crossinput = fac.Retrieve('crossinput.p', path_data)
	crossinput_sim = fac.Retrieve('crossinput_sim.p', path_data)

	X = fac.Retrieve('X.p', path_data)
	Xbar = fac.Retrieve('Xbar.p', path_data)
	Xsignals = fac.Retrieve('Xsignals.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

#


Nrows = N
Ncolumns = N+1
fac.SetPlotDim(1.9*Ncolumns, 1.7*Nrows)

ii_noise = 0

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(N):
	for ii_neuron_2 in range(N+1):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

		if ii_neuron_2 == N:
			
			plt.plot(tau_values, np.sqrt(deltaT)*crossinput[ii_neuron_1,ii_noise], color = '0')
			plt.plot(tau_values_sim, crossinput_sim[ii_neuron_1,0,:], color='r', linewidth=0.5, label='sim')

		elif ii_neuron_1 < N and ii_neuron_2 < N:

			plt.plot(tau_values, cross[ii_neuron_1,ii_neuron_2], color = '0')
			plt.plot(tau_values_sim, cross_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')
		# plt.title(str(ii_neuron_1)+'-'+str(ii_neuron_2))

plt.savefig(path_plot+'cross.pdf')

plt.show()


#

fac.SetPlotDim(1.65, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

ii_neuron_1 = 0
ii_neuron_2 = 2

plt.plot(tau_values_sim, cross_sim[ii_neuron_1,ii_neuron_2], color='#E5CCFF', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, np.flipud(cross[ii_neuron_1,ii_neuron_2]), color='#AAAAAA', linewidth=0.3)
plt.plot(tau_values, cross[ii_neuron_1,ii_neuron_2], color='#BC68EC')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 1.0)
plt.yticks([0, 0.5, 1.0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_EE.pdf')

plt.show()

#

fac.SetPlotDim(1.55, 0.95)

Tlim = 30

ii_neuron_1 = 0
ii_neuron_2 = 2

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Area 1')

plt.xlim(0, Tlim)

limit = 2.8
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

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), X[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Area 2')

plt.xlim(0, Tlim)

limit = 3.4
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

ii_signal_1 = 0
ii_signal_2 = 1

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xsignals[0,:int(Tlim/deltaT),ii_signal_1], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Signal 1')

plt.xlim(0, Tlim)

limit = 3.8
# limit = np.max(np.fabs(Xsignals[:int(Tlim/deltaT),ii_signal_1]))
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xsignal1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xsignals[0,:int(Tlim/deltaT),ii_signal_2], color='0.6', linewidth=0.6)

plt.xlabel(r'Time')
plt.ylabel(r'Signal 2')

plt.xlim(0, Tlim)

limit = 3.8
# limit = np.max(np.fabs(Xsignals[:int(Tlim/deltaT),ii_signal_1]))
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Xsignal2.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
