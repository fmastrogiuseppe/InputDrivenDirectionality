
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

sys.path.append('../../../CodeFM/')

import fct_facilities as fac
import fct_varies as var
import fct_theory as th

path_data = 'Data_simulate_theta/'
path_plot = 'Plots_simulate_theta/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 4
tau_m = 6

k = 1.01
w = 1.5
h = 0.


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### COMPUTE

# Inputs

theta_values = np.linspace(0, 90, 5)

doCompute = 1

if doCompute:

	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SETUP NETWORK

	# Connectivity

	W = np.array( [ [ w, -k*w, h, 0 ],\
					[ w, -k*w, h, 0 ],\
					[ h, 0, w, -k*w ],\
					[ h, 0, w, -k*w ] ] )

	lambdas, P = np.linalg.eig(W)
	Pinv = np.linalg.inv(P)

	#

	cross_values = []
	cross_input_values = []


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SCAN PARAMETERS

	for ii_theta, theta in enumerate(theta_values):

		u = np.array([ np.cos(theta/180*np.pi), np.sin(theta/180*np.pi), 0, 0 ])
		U = np.hstack([ np.reshape(u, (N,1)), np.zeros((N, N-1)) ])


		#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
		#### THEORY

		tau_values, crossbar, cross, crossbar_input, cross_input = th.ComputeCovarianceTheory(lambdas, P, Pinv, U, tau_m=tau_m, Ntau=601)

		cross_values.append(cross)
		cross_input_values.append(cross_input)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### SAVE

	fac.Store(tau_values, 'tau_values.p', path_data)

	fac.Store(cross_values, 'cross_values.p', path_data)
	fac.Store(cross_input_values, 'cross_input_values.p', path_data)

	fac.Store(cross_sim_values, 'cross_sim_values.p', path_data)

else:

	tau_values = fac.Retrieve('tau_values.p', path_data)

	cross_values = fac.Retrieve('cross_values.p', path_data)
	cross_input_values = fac.Retrieve('cross_input_values.p', path_data)

	cross_sim_values = fac.Retrieve('cross_sim_values.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

cm_subsection = np.linspace(0.15, 0.75, len(theta_values)) 
clr = [ cm.coolwarm_r(x) for x in cm_subsection ]

ii_noise = 0

#

Nrows = 2
Ncolumns = 2+1  # Last neuron is the input source
fac.SetPlotDim(1.65*Ncolumns, 1.5*Nrows)

fg = plt.figure()
ax = plt.axes(frameon=True)

for ii_theta, uI in enumerate(theta_values):

	for ii_neuron_1 in [0,1]:
		for ii_neuron_2 in [0,1,2]:

			plt.subplot(Nrows, Ncolumns, int(ii_neuron_1*Ncolumns+ii_neuron_2+1))

			if ii_neuron_2 == 2:
				
				plt.axvline(x=0, color='0', linewidth=0.7)
				plt.plot(tau_values, cross_input_values[ii_theta][ii_neuron_1,ii_noise], color = clr[ii_theta])

				plt.ylim(-0.6, 1.2)
				plt.yticks([-0.6, 0, 0.6, 1.2])
			
			elif ii_neuron_1 < 2 and ii_neuron_2 < 2:

				plt.axvline(x=0, color='0', linewidth=0.7)
				plt.plot(tau_values, cross_values[ii_theta][ii_neuron_1,ii_neuron_2], color = clr[ii_theta])
				# plt.plot(tau_values, np.flipud(cross_values[ii_theta][ii_neuron_1,ii_neuron_2]), color = clr[ii_theta], ls='--', linewidth=0.5)

				plt.ylim(-0.1, 1.8)
				plt.yticks([0, 0.9, 1.8])
			
			plt.xlabel(r'Lag $\tau$')
			plt.ylabel(r'Cross-covariance')

			plt.xlim(-tau_m, tau_m)
			plt.xticks([-tau_m, 0, tau_m])

plt.savefig(path_plot+'firstarea.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
