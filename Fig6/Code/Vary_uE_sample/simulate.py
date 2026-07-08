
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

path_data = 'Data_simulate/'
path_plot = 'Plots_simulate/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 8
tau_m = 7

k = 1.01
w = 1.5
h = 0.5

epsilon_w = 0.2
epsilon_h = 0.3

phi = 0.99


# Simulations

Ntrials = 100
T = 80
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 20


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

doCompute = 1

if doCompute:

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

	print(lambdas)


	# Inputs

	uI = 1
	uE1 = 3.2
	uE2 = 0.8

	u = 0.5 * np.array([ (1+phi)*uE1, (1+phi)*uI, (1-phi)*uE1, (1-phi)*uI, (1+phi)*uE2, (1+phi)*uI, (1-phi)*uE2, (1-phi)*uI ])
	U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### THEORY

	tau_values, cross_bar, cross, crossinput_bar, crossinput = \
		th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m)

	cross_latent = th.RotateCovarianceTheory(cross, R)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SIMULATIONS

	X0 = np.random.normal(0, 1, (Ntrials, N))

	X, Y, eta = var.SimulateActivity(t, X0, W, U, return_noise=1)

	X = X[:,int(Tcut/deltaT):,:]
	Y = Y[:,int(Tcut/deltaT):,:]
	eta = eta[:,int(Tcut/deltaT):,:]

	#

	Ylatent = np.swapaxes(np.reshape(np.dot(R, np.reshape(np.swapaxes(Y, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(R.shape[0], X.shape[1], X.shape[0]) ), 0, 2)
	Xlatent = np.swapaxes(np.reshape(np.dot(R, np.reshape(np.swapaxes(X, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(R.shape[0], X.shape[1], X.shape[0]) ), 0, 2)

	#

	Ybar = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(Y, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(X.shape[2], X.shape[1], X.shape[0]) ), 0, 2)
	Xbar = np.swapaxes(np.reshape(np.dot(Pinv, np.reshape(np.swapaxes(X, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(X.shape[2], X.shape[1], X.shape[0]) ), 0, 2)

	#

	cross_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))
	crossinput_sim = np.zeros(( X.shape[2], X.shape[2], len(var.ComputeCovariance(X[:,:,0], X[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(X.shape[2]):
		for ii_neuron_2 in range(X.shape[2]):

			cross_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], X[:,:,ii_neuron_2], int(tau_m/deltaT))
			crossinput_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(X[:,:,ii_neuron_1], eta[:,:,ii_neuron_2], int(tau_m/deltaT))

	tau_values_sim = np.linspace(-tau_m, tau_m, len(cross_sim[0,1]))

	cross_latent_sim = np.zeros(( Xlatent.shape[2], Xlatent.shape[2], len(var.ComputeCovariance(Xlatent[:,:,0], Xlatent[:,:,0], int(tau_m/deltaT))) ))

	for ii_neuron_1 in range(Xlatent.shape[2]):
		for ii_neuron_2 in range(Xlatent.shape[2]):

			cross_latent_sim[ii_neuron_1, ii_neuron_2,:] = var.ComputeCovariance(Xlatent[:,:,ii_neuron_1], Xlatent[:,:,ii_neuron_2], int(tau_m/deltaT))


	### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)
	fac.Store(tau_values_sim, 'tau_values_sim.p', path_data)

	fac.Store(cross, 'cross.p', path_data)
	fac.Store(cross_sim, 'cross_sim.p', path_data)
	fac.Store(crossinput, 'crossinput.p', path_data)
	fac.Store(crossinput_sim, 'crossinput_sim.p', path_data)
	fac.Store(cross_latent, 'cross_latent.p', path_data)
	fac.Store(cross_latent_sim, 'cross_latent_sim.p', path_data)

	fac.Store(Y, 'Y.p', path_data)
	fac.Store(X, 'X.p', path_data)

	fac.Store(Ylatent, 'Ylatent.p', path_data)
	fac.Store(Xlatent, 'Xlatent.p', path_data)

	fac.Store(Ybar, 'Ybar.p', path_data)
	fac.Store(Xbar, 'Xbar.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)
	tau_values_sim = fac.Retrieve('tau_values_sim.p', path_data)

	cross = fac.Retrieve('cross.p', path_data)
	cross_sim = fac.Retrieve('cross_sim.p', path_data)
	crossinput = fac.Retrieve('crossinput.p', path_data)
	crossinput_sim = fac.Retrieve('crossinput_sim.p', path_data)
	cross_latent = fac.Retrieve('cross_latent.p', path_data)
	cross_latent_sim = fac.Retrieve('cross_latent_sim.p', path_data)

	Y = fac.Retrieve('Y.p', path_data)
	X = fac.Retrieve('X.p', path_data)

	Ylatent = fac.Retrieve('Ylatent.p', path_data)
	Xlatent = fac.Retrieve('Xlatent.p', path_data)

	Ybar = fac.Retrieve('Ybar.p', path_data)
	Xbar = fac.Retrieve('Xbar.p', path_data)


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

Nrows = Nlatents
Ncolumns = Nlatents+1
fac.SetPlotDim(1.9*Ncolumns, 1.7*Nrows)

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_neuron_1 in range(Nlatents):
	for ii_neuron_2 in range(Nlatents):

		plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

		plt.plot(tau_values, cross_latent[ii_neuron_1,ii_neuron_2], color = '0')
		plt.plot(tau_values, np.flipud(cross_latent[ii_neuron_1,ii_neuron_2]), color = '0', ls='--', linewidth=0.5)
		plt.plot(np.linspace(-tau_m, tau_m, len(cross_sim[ii_neuron_1,ii_neuron_2,:])), cross_latent_sim[ii_neuron_1,ii_neuron_2,:], color='r', linewidth=0.5, label='sim')
		plt.plot(np.linspace(-tau_m, tau_m, len(cross_sim[ii_neuron_1,ii_neuron_2,:])), np.flipud(cross_latent_sim[ii_neuron_1,ii_neuron_2,:]), ls='--', color='r', linewidth=0.5, label='sim')

		plt.xlabel(r'Lag $\tau$')
		plt.ylabel(r'Cross-covariance')
		# plt.title(str(ii_neuron_1)+'-'+str(ii_neuron_2))

plt.savefig(path_plot+'cross_latent.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Cross-covariances

fac.SetPlotDim(1.65, 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

ii_neuron_1 = 0
ii_neuron_2 = 2

plt.plot(tau_values_sim, cross_latent_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, np.flipud(cross_latent[ii_neuron_1,ii_neuron_2]), color='#AAAAAA', linewidth=0.5)
plt.plot(tau_values, cross_latent[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 4.6)
plt.yticks([0, 2.3, 4.6])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_unspunsp.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

ii_neuron_1 = 1
ii_neuron_2 = 3

plt.plot(tau_values_sim, cross_latent_sim[ii_neuron_1,ii_neuron_2], color='#FFBAD2', linewidth = 4)

plt.axvline(x=0, color='0', ls='-', linewidth=0.7)

plt.plot(tau_values, np.flipud(cross_latent[ii_neuron_1,ii_neuron_2]), color='#AAAAAA', linewidth=0.5)
plt.plot(tau_values, cross_latent[ii_neuron_1,ii_neuron_2], color='#FF1A66')

plt.xlabel(r'Lag $\tau$')
plt.ylabel(r'Cross-covariance')

plt.xlim(-tau_m, tau_m)
plt.xticks([-tau_m, 0, tau_m])

plt.ylim(0, 4.6)
plt.yticks([0, 2.3, 4.6])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'cross_spsp.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Single traces plots

fac.SetPlotDim(1.45, 0.95)

Tlim = 30

ii_neuron_1 = 0
ii_neuron_2 = 1

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xlatent[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unspecif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 6.5
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X1_area1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xlatent[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Specif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 6.5
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X2_area1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ylatent[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unspecif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 7.5
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y1_area1.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ylatent[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Specif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 7.5
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y2_area1.pdf')

plt.show()

#

ii_neuron_1 = 2
ii_neuron_2 = 3

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xlatent[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unspecif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 3.8
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X1_area2.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Xlatent[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Specif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 2.4
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'X2_area2.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ylatent[0,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Unspecif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 2.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y1_area2.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(np.linspace(0, Tlim, int(Tlim/deltaT)), Ylatent[0,:int(Tlim/deltaT),ii_neuron_2], color='0.6', linewidth=0.7)

plt.xlabel(r'Time')
plt.ylabel(r'Specif')

plt.xlim(0, Tlim)
plt.xticks([ 0, 15, 30 ])

limit = 2.2
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.savefig(path_plot+'Y2_area2.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Compute PLS

Y_1 = np.reshape(Y[:,:,[0,2]], (X.shape[0]*X.shape[1], 2))
Y_2 = np.reshape(Y[:,:,[4,6]], (X.shape[0]*X.shape[1], 2))

X_1 = np.reshape(X[:,:,[0,2]], (X.shape[0]*X.shape[1], 2))
X_2 = np.reshape(X[:,:,[4,6]], (X.shape[0]*X.shape[1], 2))

#

pls = PLSRegression(n_components=1, scale=False)
pls.fit(X_1, X_2)
v1 = pls.x_weights_
v2 = pls.y_weights_


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Plane plots

ii_neuron_1 = 0
ii_neuron_2 = 1
scale = 1e3

Tlim = 400
Ntrialslim = 3

fac.SetPlotDim(0.93, 0.87)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot([0,0], [-1e3,1e3], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [0,0], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [-1e3,1e3], ls='--', color='k', linewidth=0.7)
plt.plot(Xlatent[0:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2], Xlatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.plot([-scale*np.dot(R_red, v1)[1,0], scale*np.dot(R_red, v1)[1,0]], [-scale*np.dot(R_red, v1)[0,0], scale*np.dot(R_red, v1)[0,0]], '#FF8B00')
# plt.plot([-scale*np.dot(R_red, v1)[1,1], scale*np.dot(R_red, v1)[1,1]], [-scale*np.dot(R_red, v1)[0,1], scale*np.dot(R_red, v1)[0,1]], '#FF9100', ls='--')

plt.axis('off')

limit = 12
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])
plt.xlim(-limit, limit)
plt.xticks([-limit, limit])

plt.savefig(path_plot+'X1X2_area1.pdf')
plt.savefig(path_plot+'X1X2_area1.png', dpi=300)

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot([0,0], [-1e3,1e3], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [0,0], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [-1e3,1e3], ls='--', color='k', linewidth=0.7)
plt.plot(Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2]+np.random.normal(0,0.1,Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2].shape), \
		 Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1]+np.random.normal(0,0.1,Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1].shape),\
		 color='0.6', linewidth=0.7)

plt.axis('off')

limit = 12
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])
plt.xlim(-limit, limit)
plt.xticks([-limit, limit])

plt.savefig(path_plot+'Y1Y2_area1.pdf')
plt.savefig(path_plot+'Y1Y2_area1.png', dpi=300)

plt.show()

#

ii_neuron_1 = 2
ii_neuron_2 = 3

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot([0,0], [-1e3,1e3], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [0,0], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [-1e3,1e3], ls='--', color='k', linewidth=0.7)
plt.plot(Xlatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2], Xlatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1], color='0.6', linewidth=0.7)

plt.plot([-scale*np.dot(R_red, v2)[1,0], scale*np.dot(R_red, v2)[1,0]], [-scale*np.dot(R_red, v2)[0,0], scale*np.dot(R_red, v2)[0,0]], '#FF8B00')
# plt.plot([-scale*np.dot(R_red, v2)[1,1], scale*np.dot(R_red, v2)[1,1]], [-scale*np.dot(R_red, v2)[0,1], scale*np.dot(R_red, v2)[0,1]], '#FF9100', ls='--')

plt.axis('off')

limit = 6
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])
plt.xlim(-limit, limit)
plt.xticks([-limit, limit])

plt.savefig(path_plot+'X1X2_area2.pdf')
plt.savefig(path_plot+'X1X2_area2.png', dpi=300)

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot([0,0], [-1e3,1e3], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [0,0], ls='-', color='k', linewidth=0.7)
plt.plot([-1e3,1e3], [-1e3,1e3], ls='--', color='k', linewidth=0.7)
plt.plot(Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2]+np.random.normal(0,0.04,Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_2].shape), \
		 Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1]+np.random.normal(0,0.04,Ylatent[:Ntrialslim,:int(Tlim/deltaT),ii_neuron_1].shape),\
		 color='0.6', linewidth=0.7)

plt.axis('off')

limit = 2.6
plt.ylim(-limit, limit)
plt.yticks([-limit, limit])
plt.xlim(-limit, limit)
plt.xticks([-limit, limit])

plt.savefig(path_plot+'Y1Y2_area2.pdf')
plt.savefig(path_plot+'Y1Y2_area2.png', dpi=300)

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
