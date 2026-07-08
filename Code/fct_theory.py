import matplotlib.pyplot as plt
import numpy as np
import math


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Covariance functions

def ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m = 4, Ntau = 51):

    N = np.shape(P)[1]    
    tau_values = np.linspace(-tau_m, tau_m, Ntau)
    ind_tau_pos = np.where(tau_values >= 0)
    ind_tau_neg = np.where(tau_values < 0)
    tau_pos = tau_values[ind_tau_pos]
    tau_neg = tau_values[ind_tau_neg]
         
    SigmaBar = np.zeros((P.shape[0], P.shape[0], Ntau), dtype = 'complex_')
    Sigma = np.zeros_like(SigmaBar)
       
    SigmaBarInput = np.zeros_like(SigmaBar)
    SigmaInput = np.zeros_like(SigmaBar)
       
    # Activity covariance and activity-input covariance
    
    SigmaBar_noise = np.dot( np.dot(Pinv, np.dot(U, U.T)), Pinv.T )
    SigmaBar_noise_input = np.dot(np.dot(Pinv, U), Pinv.T)
    
    for ii_neuron_1 in range(N):
        for ii_neuron_2 in range(N):

            SigmaBar[ii_neuron_1,ii_neuron_2,ind_tau_pos] = (SigmaBar_noise[ii_neuron_1,ii_neuron_2] / (2 - lambdas[ii_neuron_1] - lambdas[ii_neuron_2])) * (np.exp((lambdas[ii_neuron_2] - 1)*(tau_pos)))
            SigmaBar[ii_neuron_1,ii_neuron_2,ind_tau_neg] = (SigmaBar_noise[ii_neuron_1,ii_neuron_2] / (2 - lambdas[ii_neuron_1] - lambdas[ii_neuron_2])) * (np.exp((lambdas[ii_neuron_1] - 1)*np.fabs((tau_neg))))
                              
            SigmaBarInput[ii_neuron_1,ii_neuron_2,ind_tau_pos] = 0
            SigmaBarInput[ii_neuron_1,ii_neuron_2,ind_tau_neg] = SigmaBar_noise_input[ii_neuron_1, ii_neuron_2] * (np.exp((lambdas[ii_neuron_1] - 1)*np.fabs((tau_neg))))

    PSigmaBar = np.tensordot(P, SigmaBar, axes = (1,0))
    PSigmaBarP = np.matmul(PSigmaBar.transpose(2,0,1), P.T) 
    Sigma = PSigmaBarP.transpose(1, 2, 0)
                
    PSigmaBarInput = np.tensordot(P, SigmaBarInput, axes = (1,0))
    PSigmaBarInputP = np.matmul(PSigmaBarInput.transpose(2,0,1), P.T) 
    SigmaInput = PSigmaBarInputP.transpose(1, 2, 0)

    return tau_values, SigmaBar, Sigma, SigmaBarInput, SigmaInput


def RotateCovarianceTheory(Sigma, R):

	SigmaTilde = np.zeros(( R.shape[0], R.shape[0], Sigma.shape[-1] ))
	for ii_tau in range(Sigma.shape[-1]):
		SigmaTilde[:,:,ii_tau] = np.dot(R, np.dot(Sigma[:,:,ii_tau], R.T))

	return SigmaTilde


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Asymmetry measures

def PrincipalLag(cross, tau_values):

	N = cross.shape[0]

	asymm = np.zeros(( N, N ))

	for ii_neuron_1 in range(N):
		for ii_neuron_2 in range(N):
			asymm[ii_neuron_1, ii_neuron_2] = tau_values[np.argmax(np.fabs(cross[ii_neuron_1, ii_neuron_2]))]
			# asymm[ii_neuron_1, ii_neuron_2] = tau_values[np.argmax((cross[ii_neuron_1, ii_neuron_2]))]

	return asymm


def AsymmetryScore(cross, tau_values):

	N = cross.shape[0]
	T = cross.shape[-1]

	asymm = np.zeros(( N, N ))

	for ii_neuron_1 in range(N):
		for ii_neuron_2 in range(N):
			asymm[ii_neuron_1, ii_neuron_2] = (tau_values[1]-tau_values[0]) \
											* ( np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values>0)[0]]) - \
												np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values<0)[0]]) )

			asymm[ii_neuron_1, ii_neuron_2] /= (tau_values[1]-tau_values[0]) \
											* ( np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values>0)[0]]) + \
												np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values<0)[0]]) )

	return asymm


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Covariance measures

def Peak(cross, tau_values):

	N = cross.shape[0]

	cov = np.zeros(( N, N ))

	for ii_neuron_1 in range(N):
		for ii_neuron_2 in range(N):
			cov[ii_neuron_1, ii_neuron_2] = np.max(np.fabs(cross[ii_neuron_1, ii_neuron_2]))

	return cov


def TotalAUC(cross, tau_values):

	N = cross.shape[0]
	T = cross.shape[-1]

	cov = np.zeros(( N, N ))

	for ii_neuron_1 in range(N):
		for ii_neuron_2 in range(N):
			cov[ii_neuron_1, ii_neuron_2] = (tau_values[1]-tau_values[0]) \
											* ( np.fabs(np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values>0)[0]])) + \
												np.fabs(np.sum(cross[ii_neuron_1, ii_neuron_2,np.where(tau_values<0)[0]])) )

	return cov


def Variance(cross, tau_values):

	N = cross.shape[0]
	T = cross.shape[-1]

	cov = np.zeros(( N, N ))

	for ii_neuron_1 in range(N):
		for ii_neuron_2 in range(N):
			cov[ii_neuron_1, ii_neuron_2] = cross[ii_neuron_1, ii_neuron_1, math.floor(int(T)*0.5)]

	return cov

