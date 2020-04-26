import os
import numpy as np
import math
from montepython.likelihood_class import Likelihood_prior
from time import time


class euclid_B1loop(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

		h=cosmo.h()
		t1=time()
		#get ntriag
		triag = [ [0 for x in range(self.ksize*self.ksize*self.ksize)] for y in range(3)]

		ntriag = 0
		for i1 in range(self.ksize):
			for i2 in range(i1+1):
				if i1-i2 + 1 > 1:
					istar = i1-i2
				else:
					istar = 0 
				for i3 in range(istar,i2+1):
					triag[0][ntriag] = i1
					triag[1][ntriag] = i2
					triag[2][ntriag] = i3
					ntriag = ntriag + 1
         		   #print(i1,i2,i3)

		new_triag = [ [0 for x in range(ntriag)] for y in range(3)]
		for i in range(3):
			for j in range(ntriag):
				new_triag[i][j] = triag[i][j]

		Plin = np.zeros(self.ksize, 'float64')
		self.Bk = np.zeros(ntriag,'float64')
		self.invcov = np.zeros(
            (ntriag, ntriag), 'float64')
		delta = np.zeros(ntriag,'float64')
		
		datafile = open(os.path.join(self.data_directory, self.measurements_file[0]), 'r')
		line = datafile.readlines()
		datafile.close()
		kmin = float(line[0].split()[0])
		kmax = float(line[-1].split()[0])
		k = np.zeros(self.ksize, 'float64')
		k = np.linspace(kmin,kmax,self.ksize)

		norm = (data.mcmc_parameters['norm']['current'] *
		         data.mcmc_parameters['norm']['scale'])

		b1 = np.zeros(self.zsize,'float64')
		b1[0] = (data.mcmc_parameters['b1_1']['current'] *
                 data.mcmc_parameters['b1_1']['scale'])
		b1[1] = (data.mcmc_parameters['b1_2']['current'] *
                 data.mcmc_parameters['b1_2']['scale'])
		b1[2] = (data.mcmc_parameters['b1_3']['current'] *
                 data.mcmc_parameters['b1_3']['scale'])
		b1[3] = (data.mcmc_parameters['b1_4']['current'] *
                 data.mcmc_parameters['b1_4']['scale'])
		b1[4] = (data.mcmc_parameters['b1_5']['current'] *
                 data.mcmc_parameters['b1_5']['scale'])
		b1[5] = (data.mcmc_parameters['b1_6']['current'] *
                 data.mcmc_parameters['b1_6']['scale'])
		b1[6] = (data.mcmc_parameters['b1_7']['current'] *
                 data.mcmc_parameters['b1_7']['scale'])
		b1[7] = (data.mcmc_parameters['b1_8']['current'] *
                 data.mcmc_parameters['b1_8']['scale'])

		b2 = np.zeros(self.zsize,'float64')
		b2[0] = (data.mcmc_parameters['b2_1']['current'] *
                 data.mcmc_parameters['b2_1']['scale'])
		b2[1] = (data.mcmc_parameters['b2_2']['current'] *
                 data.mcmc_parameters['b2_2']['scale'])
		b2[2] = (data.mcmc_parameters['b2_3']['current'] *
                 data.mcmc_parameters['b2_3']['scale'])
		b2[3] = (data.mcmc_parameters['b2_4']['current'] *
                 data.mcmc_parameters['b2_4']['scale'])
		b2[4] = (data.mcmc_parameters['b2_5']['current'] *
                 data.mcmc_parameters['b2_5']['scale'])
		b2[5] = (data.mcmc_parameters['b2_6']['current'] *
                 data.mcmc_parameters['b2_6']['scale'])
		b2[6] = (data.mcmc_parameters['b2_7']['current'] *
                 data.mcmc_parameters['b2_7']['scale'])
		b2[7] = (data.mcmc_parameters['b2_8']['current'] *
                 data.mcmc_parameters['b2_8']['scale'])

		bG2 = np.zeros(self.zsize,'float64')
		bG2[0] = (data.mcmc_parameters['bG2_1']['current'] *
                 data.mcmc_parameters['bG2_1']['scale'])
		bG2[1] = (data.mcmc_parameters['bG2_2']['current'] *
                 data.mcmc_parameters['bG2_2']['scale'])
		bG2[2] = (data.mcmc_parameters['bG2_3']['current'] *
                 data.mcmc_parameters['bG2_3']['scale'])
		bG2[3] = (data.mcmc_parameters['bG2_4']['current'] *
                 data.mcmc_parameters['bG2_4']['scale'])
		bG2[4] = (data.mcmc_parameters['bG2_5']['current'] *
                 data.mcmc_parameters['bG2_5']['scale'])
		bG2[5] = (data.mcmc_parameters['bG2_6']['current'] *
                 data.mcmc_parameters['bG2_6']['scale'])
		bG2[6] = (data.mcmc_parameters['bG2_7']['current'] *
                 data.mcmc_parameters['bG2_7']['scale'])
		bG2[7] = (data.mcmc_parameters['bG2_8']['current'] *
                 data.mcmc_parameters['bG2_8']['scale'])

		Pshot = np.zeros(self.zsize,'float64')
		Pshot[0] = (data.mcmc_parameters['Pshot_1']['current'] *
                 data.mcmc_parameters['Pshot_1']['scale'])
		Pshot[1] = (data.mcmc_parameters['Pshot_2']['current'] *
                 data.mcmc_parameters['Pshot_2']['scale'])
		Pshot[2] = (data.mcmc_parameters['Pshot_3']['current'] *
                 data.mcmc_parameters['Pshot_3']['scale'])
		Pshot[3] = (data.mcmc_parameters['Pshot_4']['current'] *
                 data.mcmc_parameters['Pshot_4']['scale'])
		Pshot[4] = (data.mcmc_parameters['Pshot_5']['current'] *
                 data.mcmc_parameters['Pshot_5']['scale'])
		Pshot[5] = (data.mcmc_parameters['Pshot_6']['current'] *
                 data.mcmc_parameters['Pshot_6']['scale'])
		Pshot[6] = (data.mcmc_parameters['Pshot_7']['current'] *
                 data.mcmc_parameters['Pshot_7']['scale'])
		Pshot[7] = (data.mcmc_parameters['Pshot_8']['current'] *
                 data.mcmc_parameters['Pshot_8']['scale'])

		Pshots = np.zeros(self.zsize,'float64')
		Pshots[0] = (data.mcmc_parameters['Pshots_1']['current'] *
                 data.mcmc_parameters['Pshots_1']['scale'])
		Pshots[1] = (data.mcmc_parameters['Pshots_2']['current'] *
                 data.mcmc_parameters['Pshots_2']['scale'])
		Pshots[2] = (data.mcmc_parameters['Pshots_3']['current'] *
                 data.mcmc_parameters['Pshots_3']['scale'])
		Pshots[3] = (data.mcmc_parameters['Pshots_4']['current'] *
                 data.mcmc_parameters['Pshots_4']['scale'])
		Pshots[4] = (data.mcmc_parameters['Pshots_5']['current'] *
                 data.mcmc_parameters['Pshots_5']['scale'])
		Pshots[5] = (data.mcmc_parameters['Pshots_6']['current'] *
                 data.mcmc_parameters['Pshots_6']['scale'])
		Pshots[6] = (data.mcmc_parameters['Pshots_7']['current'] *
                 data.mcmc_parameters['Pshots_7']['scale'])
		Pshots[7] = (data.mcmc_parameters['Pshots_8']['current'] *
                 data.mcmc_parameters['Pshots_8']['scale'])

		chi2 = 0.
		for z_i in range(self.zsize):
			fz = cosmo.scale_independent_growth_factor_f(self.z[z_i])
 			beta = fz/b1[z_i]
			a0 = 1. + 2.*beta/3. + beta**2./5.
			datafile = open(os.path.join(self.data_directory, self.measurements_file[z_i]), 'r')
			for i in range(ntriag):
				line = datafile.readline()
				self.Bk[i] = float(line.split()[3])
			datafile.close()

			self.invcov = np.loadtxt(os.path.join(self.data_directory, self.covmat_file[z_i]))
       
			for i in range(self.ksize):
				Plin[i] = -1.*cosmo.pk(k[i]*h, self.z[z_i])[10]/h**2./k[i]**2.

			D1 = lambda k1,k2,k3: (15. + 10.*beta+beta**2. + 2.*beta**2.*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2.)/15.
			D2 = lambda k1,k2,k3: beta*(35.*(k1/k2)**2. + 28.*beta*(k1/k2)**2. + 3.*beta**2.*(k1/k2)**2. + 35. + 28.*beta + 3.*beta**2. + 70.*(k1/k2)*(k3**2. - k1**2. - k2**2.)/(2.*k1*1.*1.*k2) + 84.*beta*(k1/k2)*(k3**2.-k1**2.-k2**2.)/(2.*k1*k2) + 18.*beta**2.*(k1/k2)*(k3**2.-k1**2.-k2**2.)/(2.*k1*k2) + 14.*beta*((k1/k2)*(k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2. + 12.*beta**2.*((k1/k2)*(k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2. + 14.*beta*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2.+ 12.*beta**2.*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2. + 12.*beta**2.*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2.*(k1/k2))/(105.*(1.+(k1/k2)**2.+2.*(k1/k2)*(k3**2.-k1**2.-k2**2.)/(2.*k1*k2)))
			F2 = lambda k1,k2,k3: (b1[z_i]*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.)*D1(k1,k2,k3) + (-3.*(k1**2.-k2**2.)**2.-1.*(k1**2.+k2**2.)*k3**2.+4.*k3**4.)*D2(k1,k2,k3) + 7.*D1(k1,k2,k3)*(2.*b2[z_i]*k1**2.*k2**2. + bG2[z_i]*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1[z_i]**2./28./k1**2./k2**2.

			for j in range(ntriag):
				delta[j] = 2.*norm**2.*F2(k[new_triag[0][j]],k[new_triag[1][j]],k[new_triag[2][j]])*Plin[new_triag[0][j]]*Plin[new_triag[1][j]]*h**6. + 2.*norm**2.*F2(k[new_triag[0][j]],k[new_triag[2][j]],k[new_triag[1][j]])*Plin[new_triag[0][j]]*Plin[new_triag[2][j]]*h**6. + 2.*norm**2.*F2(k[new_triag[1][j]],k[new_triag[2][j]],k[new_triag[0][j]])*Plin[new_triag[2][j]]*Plin[new_triag[1][j]]*h**6. + Pshot[z_i]*norm*b1[z_i]**2.*(Plin[new_triag[0][j]]+Plin[new_triag[1][j]]+Plin[new_triag[2][j]])*h**3.*a0 + Pshots[z_i] - self.Bk[j]

			chi2=chi2+np.dot(delta,np.dot(self.invcov,delta))

		#print("chi2_euclidB=", chi2)
		loglkl = -0.5 * chi2

		return loglkl
