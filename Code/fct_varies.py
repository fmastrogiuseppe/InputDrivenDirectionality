import matplotlib.pyplot as plt
import numpy as np


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Simulations

def SimulateActivity(t, X0, W, U, eta=0, target_noise=0, return_noise=0, rectify=0, bias=0): # Number of trials specified by dimensionality of X0

	print (' ** Simulating... **')

	# Setup

	deltaT = t[1]-t[0]

	X = np.zeros(( X0.shape[0], len(t), X0.shape[1] ))
	X[:,0,:] = X0

	if not target_noise:
		eta = np.random.normal(0, 1, X.shape)
	Y = np.swapaxes(np.reshape(np.dot(U, np.reshape(np.swapaxes(eta, 0, 2), (X.shape[2], X.shape[0]*X.shape[1])) ), \
		(X.shape[2], X.shape[1], X.shape[0]) ), 0, 2)

	# Integrate activity

	for ii_trial in range(X0.shape[0]):
		for ii_time in range(len(t[:len(t)-1])):

			X[ii_trial,ii_time+1,:] = X[ii_trial,ii_time,:] + deltaT*(-X[ii_trial,ii_time,:] + np.dot(W, X[ii_trial,ii_time,:])+ bias) \
								+ np.sqrt(deltaT)*Y[ii_trial,ii_time,:]

			if rectify: X[ii_time+1,X[ii_time+1,:]<0] = 0

	if return_noise: return X, Y, eta
	else: return X


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Analysis

def PCA(X, doCenter=False, doNormalize=False):

	if doCenter: X = X-np.mean(X,0)[None,:]
	if doNormalize: X = X/np.std(X,0)[None,:]

	eigvals, v = np.linalg.eig(np.dot(X.T, X)/X.shape[0])

	v = v[:,np.flipud(np.argsort(eigvals))]
	eigvals = eigvals[np.flipud(np.argsort(eigvals))]

	Xproj = (np.dot(v.T, X.T)).T

	return eigvals.real, v.real, Xproj.real


def ComputeCovariance(signalX, signalY, window): # Input data: Ntrials x T activity matrix

	print (' ** Computing cross-covariance... **')

	cov = np.nan*np.zeros(( signalX.shape[1], 2*window+1 ))

	for ii_time in range(signalX.shape[1]):

		for ii_delay, delay in enumerate(np.arange(-window, window+1)):

			ind_init_X = ii_time
			ind_end_X = ii_time+1
			X = np.squeeze(signalX[:,ind_init_X:ind_end_X])

			ind_init_Y = ii_time+delay
			ind_end_Y = ii_time+delay+1
			Y = np.squeeze(signalY[:,ind_init_Y:ind_end_Y])

			if (ind_init_X) >=0 and (ind_init_Y) >= 0 and (ind_end_X) <= signalX.shape[1] and (ind_end_Y) <= signalX.shape[1]:
				cov[ii_time,ii_delay] = np.cov(X, Y)[0,1]

	return np.nanmean(cov, 0)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Utilities

def Normalize(vector):
	return vector/np.sqrt(np.sum(vector**2))


def OrthogonalBasis(M, N):

	z = np.random.normal(0, 1, (M,N))    # from Gaussian statistics
	z[0,:] = Normalize(z[0,:])

	for ii_vector in range(M-1):    # apply ortho-normalization
		for ii_past in range(ii_vector+1):

			z[ii_vector+1,:] -= np.dot(z[ii_vector+1,:],z[ii_vector-ii_past,:])*z[ii_vector-ii_past,:]
		
		z[ii_vector+1,:] = Normalize(z[ii_vector+1,:])

	return z


def RandomConnectivity(N, exclude_imag=True, lambda_low=1e8, lambda_high=-1e8, rescale_to=0.5):

	loop = 1

	while loop:

		W = np.random.normal(0, 0.5, (N,N))
		lambdas, P = np.linalg.eig(W)

		if ( (np.max(lambdas)>0 and np.max(lambdas)<1 and len(np.where(lambdas.imag==0)[0])==N) and \
			 (np.min(lambdas)<lambda_low and np.max(lambdas)>lambda_high) ):
			loop = 0

	W = W/np.max(lambdas)*rescale_to
	lambdas = lambdas/np.max(lambdas)*rescale_to

	P = P[:,np.flipud(np.argsort(lambdas))]
	lambdas = lambdas[np.flipud(np.argsort(lambdas))]

	return W, lambdas, P


def ShuffleConnectivity(W, rescale_to=0.5):

	W0 = np.copy(W)

	loop = 1

	while loop:

		np.random.shuffle(W)
		lambdas, P = np.linalg.eig(W)

		if not np.array_equal(W, W0):
			if (np.max(lambdas)>0 and np.max(lambdas)<1 and len(np.where(lambdas.imag==0)[0])==len(lambdas)):
				loop = 0

	lambdas, P = np.linalg.eig(W)

	W = W/np.max(lambdas)*rescale_to
	lambdas = lambdas/np.max(lambdas)*rescale_to

	P = P[:,np.flipud(np.argsort(lambdas))]
	lambdas = lambdas[np.flipud(np.argsort(lambdas))]

	return W, lambdas, P


def ParticipationRatio(vector):
	return np.sum(vector)**2 / (np.sum(vector**2))


def WidthHalfHeight(signal): # Input data: many cross-covariance functions

	tau = np.zeros(signal.shape[0])

	for ii in range(signal.shape[0]):

		# Find left pivot
		for ii_lag in range(len(signal[0,:])-1):
			if signal[ii,ii_lag+1]>=0.5*np.max(signal[ii,:]): 
				ii_left = ii_lag
				break

		# Find right pivot
		for ii_lag in range(len(signal[0,:])-1):
			if signal[ii,signal.shape[1]-1-ii_lag]>=0.5*np.max(signal[ii,:]): 
				ii_right = signal.shape[1]-ii_lag
				break

		tau[ii] = ii_right-ii_left

	return tau


def Bin(data, bin_size): # Input data: Ntrials x T x N activity matrix

	# Cut final time points you can't bin

	data = data[:,:(data.shape[1]-data.shape[1]%bin_size),:]

	# Bin

	data_binned = np.zeros_like(data)

	for ii_trial in range(data.shape[0]):
		for ii_neuron in range(data.shape[2]):

			data_binned[ii_trial,:,ii_neuron] = np.repeat(np.mean(np.reshape(data[ii_trial,:,ii_neuron], \
										(int(data.shape[1]/float(bin_size)), bin_size)), 1), bin_size)

	return data_binned

