import os
import numpy as np
import math
from montepython.likelihood_class import Likelihood_prior
from time import time


class euclid_P1loopAP(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        self.k = np.zeros(self.ksize,'float64')
        self.Pk0 = np.zeros(self.ksize,'float64')
        self.Pk2 = np.zeros(self.ksize,'float64')
        self.Pk4 = np.zeros(self.ksize,'float64')

        self.invcov = np.zeros(
            (3*self.ksize, 3*self.ksize), 'float64')

        delta = np.zeros(3*self.ksize,'float64')

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

        bGamma3 = np.zeros(self.zsize,'float64')
        bGamma3[0] = 0.076666666666666744
        bGamma3[1] = 0.12047619047619047
        bGamma3[2] = 0.16428571428571431
        bGamma3[3] = 0.20809523809523806
        bGamma3[4] = 0.25190476190476191
        bGamma3[5] = 0.29571428571428576
        bGamma3[6] = 0.33952380952380962
        bGamma3[7] = 0.38333333333333347

        css0 = np.zeros(self.zsize,'float64')
        css0[0] = (data.mcmc_parameters['css0_1']['current'] *
                 data.mcmc_parameters['css0_1']['scale'])
        css0[1] = (data.mcmc_parameters['css0_2']['current'] *
                 data.mcmc_parameters['css0_2']['scale'])
        css0[2] = (data.mcmc_parameters['css0_3']['current'] *
                 data.mcmc_parameters['css0_3']['scale'])
        css0[3] = (data.mcmc_parameters['css0_4']['current'] *
                 data.mcmc_parameters['css0_4']['scale'])
        css0[4] = (data.mcmc_parameters['css0_5']['current'] *
                 data.mcmc_parameters['css0_5']['scale'])
        css0[5] = (data.mcmc_parameters['css0_6']['current'] *
                 data.mcmc_parameters['css0_6']['scale'])
        css0[6] = (data.mcmc_parameters['css0_7']['current'] *
                 data.mcmc_parameters['css0_7']['scale'])
        css0[7] = (data.mcmc_parameters['css0_8']['current'] *
                 data.mcmc_parameters['css0_8']['scale'])

        css2 = np.zeros(self.zsize,'float64')
        css2[0] = (data.mcmc_parameters['css2_1']['current'] *
                 data.mcmc_parameters['css2_1']['scale'])
        css2[1] = (data.mcmc_parameters['css2_2']['current'] *
                 data.mcmc_parameters['css2_2']['scale'])
        css2[2] = (data.mcmc_parameters['css2_3']['current'] *
                 data.mcmc_parameters['css2_3']['scale'])
        css2[3] = (data.mcmc_parameters['css2_4']['current'] *
                 data.mcmc_parameters['css2_4']['scale'])
        css2[4] = (data.mcmc_parameters['css2_5']['current'] *
                 data.mcmc_parameters['css2_5']['scale'])
        css2[5] = (data.mcmc_parameters['css2_6']['current'] *
                 data.mcmc_parameters['css2_6']['scale'])
        css2[6] = (data.mcmc_parameters['css2_7']['current'] *
                 data.mcmc_parameters['css2_7']['scale'])
        css2[7] = (data.mcmc_parameters['css2_8']['current'] *
                 data.mcmc_parameters['css2_8']['scale'])

        css4 = np.zeros(self.zsize,'float64')
        css4[0] = (data.mcmc_parameters['css4_1']['current'] *
                 data.mcmc_parameters['css4_1']['scale'])
        css4[1] = (data.mcmc_parameters['css4_2']['current'] *
                 data.mcmc_parameters['css4_2']['scale'])
        css4[2] = (data.mcmc_parameters['css4_3']['current'] *
                 data.mcmc_parameters['css4_3']['scale'])
        css4[3] = (data.mcmc_parameters['css4_4']['current'] *
                 data.mcmc_parameters['css4_4']['scale'])
        css4[4] = (data.mcmc_parameters['css4_5']['current'] *
                 data.mcmc_parameters['css4_5']['scale'])
        css4[5] = (data.mcmc_parameters['css4_6']['current'] *
                 data.mcmc_parameters['css4_6']['scale'])
        css4[6] = (data.mcmc_parameters['css4_7']['current'] *
                 data.mcmc_parameters['css4_7']['scale'])
        css4[7] = (data.mcmc_parameters['css4_8']['current'] *
                 data.mcmc_parameters['css4_8']['scale'])

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

        t1 = time()

        h = cosmo.h()
        chi2 = 0.
        for z_i in range(self.zsize):

            fz = cosmo.scale_independent_growth_factor_f(self.z[z_i])
            datafile = open(os.path.join(self.data_directory, self.measurements_file[z_i]), 'r')
            for i in range(self.ksize):
                line = datafile.readline()
                self.k[i] = float(line.split()[0])
                self.Pk0[i] = float(line.split()[1])
                self.Pk2[i] = float(line.split()[2])
                self.Pk4[i] = float(line.split()[3])
            datafile.close()

            self.invcov = np.loadtxt(os.path.join(self.data_directory, self.covmat_file[z_i]))

            for i in range(self.ksize):
                kinloop1 = self.k[i]*h
                delta[2*self.ksize + i] = (norm*cosmo.pk(kinloop1, self.z[z_i])[20] +norm**2.*(cosmo.pk(kinloop1, self.z[z_i])[27])+ b1[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[28] + b1[z_i]**2.*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[29] + b2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[38] + bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[39] + 2.*css4[z_i]*norm*cosmo.pk(kinloop1, self.z[z_i])[13]/h**2.)*h**3. - self.Pk4[i]
 
                delta[self.ksize + i] = (norm*cosmo.pk(kinloop1, self.z[z_i])[18] +norm**2.*(cosmo.pk(kinloop1, self.z[z_i])[24])+ norm*b1[z_i]*cosmo.pk(kinloop1, self.z[z_i])[19] +norm**2.*b1[z_i]*(cosmo.pk(kinloop1, self.z[z_i])[25]) + b1[z_i]**2.*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[26] +b1[z_i]*b2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[34]+ b2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[35] + b1[z_i]*bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[36]+ bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[37] + (2.*b1[z_i]**2./3.+fz*b1[z_i]*8./7.+fz**2.*10./21.)*fz**1.*(3./2.)*css2[z_i]*norm*cosmo.pk(kinloop1, self.z[z_i])[12]/h**2. + (2.*bG2[z_i]+0.8*bGamma3[z_i])*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[9])*h**3. - self.Pk2[i]

                delta[i] = (norm*cosmo.pk(kinloop1, self.z[z_i])[15] +norm**2.*(cosmo.pk(kinloop1, self.z[z_i])[21])+ norm*b1[z_i]*cosmo.pk(kinloop1, self.z[z_i])[16] +norm**2.*b1[z_i]*(cosmo.pk(kinloop1, self.z[z_i])[22]) + norm*b1[z_i]**2.*cosmo.pk(kinloop1, self.z[z_i])[17] +norm**2.*b1[z_i]**2.*(cosmo.pk(kinloop1, self.z[z_i])[23]) + 0.25*norm**2.*b2[z_i]**2.*cosmo.pk(kinloop1, self.z[z_i])[1] +b1[z_i]*b2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[30]+ b2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[31] + b1[z_i]*bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[32]+ bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[33]+b2[z_i]*bG2[z_i]*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[4]+ bG2[z_i]**2.*norm**2.*cosmo.pk(kinloop1, self.z[z_i])[5] + css0[z_i]*(b1[z_i]**2./3.+fz*b1[z_i]*2./5.+fz**2./7.)*fz**2.*norm*cosmo.pk(kinloop1, self.z[z_i])[11]/h**2. + (2.*bG2[z_i]+0.8*bGamma3[z_i])*norm**2.*(b1[z_i]*cosmo.pk(kinloop1, self.z[z_i])[7]+cosmo.pk(kinloop1, self.z[z_i])[8]))*h**3. + Pshot[z_i] - self.Pk0[i]

            chi2=chi2+np.dot(delta,np.dot(self.invcov,delta))

        #print("chi2_euclidP=", chi2)
        loglkl = -0.5 * chi2

        return loglkl
