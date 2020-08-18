import os
import numpy as np
from montepython.likelihood_class import Likelihood_prior

from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from scipy.special import gamma
from scipy import interpolate
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special

class sgc_z1_new(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.

    def __init__(self,path,data,command_line):

        Likelihood_prior.__init__(self,path,data,command_line)

        # First load in data

        self.k = np.zeros(self.ksize,'float64')
        self.Pk0 = np.zeros(self.ksize,'float64')
        self.Pk2 = np.zeros(self.ksize,'float64')

        self.invcov = np.zeros(
            (2*self.ksize, 2*self.ksize), 'float64')
        cov = np.zeros(
            (2*self.ksize, 2*self.ksize), 'float64')

        datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
        for i in range(2*self.ksize):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            for j in range(2*self.ksize):
                cov[i,j] = float(line.split()[j])
        self.invcov = np.linalg.inv(cov)
        datafile.close()

        datafile = open(os.path.join(self.data_directory, self.measurements_file), 'r')
        for i in range(self.ksize):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.k[i] = float(line.split()[0])
            self.Pk0[i] = float(line.split()[1])
            self.Pk2[i] = float(line.split()[2])

        datafile.close()

        self.Nmax=128
        self.W0 = np.zeros(self.Nmax)
        self.W2 = np.zeros(self.Nmax)
        self.W4 = np.zeros(self.Nmax)
        datafile = open(os.path.join(self.data_directory, self.window_file), 'r')
        for i in range(self.Nmax):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.W0[i] = float(line.split()[0])
            self.W2[i] = float(line.split()[1])
            self.W4[i] = float(line.split()[2])
        datafile.close()

        # Precompute useful window function things
        kmax = 100.
        self.k0 = 1.e-4

        self.rmin = 0.01
        rmax = 1000.
        b = -1.1001
        bR = -2.001

        Delta = log(kmax/self.k0) / (self.Nmax - 1)
        Delta_r = log(rmax/self.rmin) / (self.Nmax - 1)
        i_arr = np.arange(self.Nmax)
        rtab = self.rmin * exp(Delta_r * i_arr)

        self.kbins3 = self.k0 * exp(Delta * i_arr)
        self.tmp_factor = exp(-1.*b*i_arr*Delta)
        self.tmp_factor2 = exp(-1.*bR*i_arr*Delta_r)

        jsNm = np.arange(-self.Nmax/2,self.Nmax/2+1,1)
        self.etam = b + 2*1j*pi*(jsNm)/self.Nmax/Delta

        def J_func(r,nu):
            gam = special.gamma(2+nu)
            r_pow = r**(-3.-1.*nu)
            sin_nu = np.sin(pi*nu/2.)
            J0 = -1.*sin_nu*r_pow*gam/(2.*pi**2.)
            J2 = r_pow*(3.+nu)*gam*sin_nu/(nu*2.*pi**2.)
            return J0,J2

        self.J0_arr,self.J2_arr = J_func(rtab.reshape(-1,1),self.etam.reshape(1,-1))

        self.etamR = bR + 2*1j*pi*(jsNm)/self.Nmax/Delta_r

        def Jk_func(k,nu):
            gam = special.gamma(2+nu)
            k_pow = k**(-3.-1.*nu)
            sin_nu = np.sin(pi*nu/2.)
            J0k = -1.*k_pow*gam*sin_nu*(4.*pi)
            J2k = k_pow*(3.+nu)*gam*sin_nu*4.*pi/nu
            return J0k,J2k

        self.J0k_arr,self.J2k_arr = Jk_func(self.kbins3.reshape(-1,1),self.etamR.reshape(1,-1))


    def loglkl(self, cosmo, data):

        h = cosmo.h()

        norm = (data.mcmc_parameters['norm']['current'] *
                 data.mcmc_parameters['norm']['scale'])
        i_s=repr(4)
        b1 = (data.mcmc_parameters['b^{('+i_s+')}_1']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_1']['scale'])
        b2 = (data.mcmc_parameters['b^{('+i_s+')}_2']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_2']['scale'])
        bG2 = (data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['scale'])
        css0 = (data.mcmc_parameters['c^{('+i_s+')}_{0}']['current'] *
             data.mcmc_parameters['c^{('+i_s+')}_{0}']['scale'])
        css2 = (data.mcmc_parameters['c^{('+i_s+')}_{2}']['current'] *
             data.mcmc_parameters['c^{('+i_s+')}_{2}']['scale'])
        Pshot = (data.mcmc_parameters['P^{('+i_s+')}_{shot}']['current'] *
             data.mcmc_parameters['P^{('+i_s+')}_{shot}']['scale'])
        b4 = (data.mcmc_parameters['b^{('+i_s+')}_4']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_4']['scale'])

        bGamma3 = 0.
        a2 = 0.
        Nmax = self.Nmax
        k0 = self.k0

        z = self.z;
        fz = cosmo.scale_independent_growth_factor_f(z)

        theory0 = np.zeros(Nmax)
        theory2 = np.zeros(Nmax)
        for i in range(Nmax):
            kinloop1 = self.kbins3[i] * h
            theory2[i] = (norm**2.*cosmo.pk(kinloop1, self.z)[18] +norm**4.*(cosmo.pk(kinloop1, self.z)[24])+ norm**1.*b1*cosmo.pk(kinloop1, self.z)[19] +norm**3.*b1*(cosmo.pk(kinloop1, self.z)[25]) + b1**2.*norm**2.*cosmo.pk(kinloop1, self.z)[26] +b1*b2*norm**2.*cosmo.pk(kinloop1, self.z)[34]+ b2*norm**3.*cosmo.pk(kinloop1, self.z)[35] + b1*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[36]+ bG2*norm**3.*cosmo.pk(kinloop1, self.z)[37]  + 2.*(css2 + 0.*b4*kinloop1**2.)*norm**2.*cosmo.pk(kinloop1, self.z)[12]/h**2. + (2.*bG2+0.8*bGamma3)*norm**3.*cosmo.pk(kinloop1, self.z)[9])*h**3. + fz**2.*b4*self.kbins3[i]**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*cosmo.pk(kinloop1, self.z)[13]*h
            theory0[i] = (norm**2.*cosmo.pk(kinloop1, self.z)[15] +norm**4.*(cosmo.pk(kinloop1, self.z)[21])+ norm**1.*b1*cosmo.pk(kinloop1, self.z)[16] +norm**3.*b1*(cosmo.pk(kinloop1, self.z)[22]) + norm**0.*b1**2.*cosmo.pk(kinloop1, self.z)[17] +norm**2.*b1**2.*cosmo.pk(kinloop1, self.z)[23] + 0.25*norm**2.*b2**2.*cosmo.pk(kinloop1, self.z)[1] +b1*b2*norm**2.*cosmo.pk(kinloop1, self.z)[30]+ b2*norm**3.*cosmo.pk(kinloop1, self.z)[31] + b1*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[32]+ bG2*norm**3.*cosmo.pk(kinloop1, self.z)[33] + b2*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[4]+ bG2**2.*norm**2.*cosmo.pk(kinloop1, self.z)[5] + 2.*css0*norm**2.*cosmo.pk(kinloop1, self.z)[11]/h**2. + (2.*bG2+0.8*bGamma3)*norm**2.*(b1*cosmo.pk(kinloop1, self.z)[7]+norm*cosmo.pk(kinloop1, self.z)[8]))*h**3.+Pshot + fz**2.*b4*self.kbins3[i]**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*cosmo.pk(kinloop1, self.z)[13]*h

        i_arr = np.arange(Nmax)
        factor = exp(-1.*(self.kbins3*h/2.)**4.)*self.tmp_factor
        Pdiscrin0 = theory0*factor
        Pdiscrin2 = theory2*factor

        cm0 = np.fft.fft(Pdiscrin0)/ Nmax
        cm2 = np.fft.fft(Pdiscrin2)/ Nmax
        cmsym0 = np.zeros(Nmax+1,dtype=np.complex_)
        cmsym2 = np.zeros(Nmax+1,dtype=np.complex_)

        all_i = np.arange(Nmax+1)
        f = (all_i+2-Nmax/2) < 1
        cmsym0[f] = k0**(-self.etam[f])*np.conjugate(cm0[-all_i[f]+Nmax/2])
        cmsym2[f] = k0**(-self.etam[f])*np.conjugate(cm2[-all_i[f]+Nmax/2])
        cmsym0[~f] = k0**(-self.etam[~f])*cm0[all_i[~f]-Nmax/2]
        cmsym2[~f] = k0**(-self.etam[~f])*cm2[all_i[~f]-Nmax/2]

        cmsym0[-1] = cmsym0[-1] / 2
        cmsym0[0] = cmsym0[0] / 2
        cmsym2[-1] = cmsym2[-1] / 2
        cmsym2[0] = cmsym2[0] / 2

        xi0 = np.real(cmsym0*self.J0_arr).sum(axis=1)
        xi2 = np.real(cmsym2*self.J2_arr).sum(axis=1)

        i_arr = np.arange(Nmax)
        Xidiscrin0 = (xi0*self.W0 + 0.2*xi2*self.W2)*self.tmp_factor2
        Xidiscrin2 = (xi0*self.W2 + xi2*(self.W0 + 2.*(self.W2+self.W4)/7.))*self.tmp_factor2

        cmr0 = np.fft.fft(Xidiscrin0)/ Nmax
        cmr2 = np.fft.fft(Xidiscrin2)/ Nmax

        cmsymr0 = np.zeros(Nmax+1,dtype=np.complex_)
        cmsymr2 = np.zeros(Nmax+1,dtype=np.complex_)

        arr_i = np.arange(Nmax+1)
        f = (arr_i+2-Nmax/2)<1
        cmsymr0[f] = self.rmin**(-self.etamR[f])*np.conjugate(cmr0[-arr_i[f] + Nmax/2])
        cmsymr2[f] =  self.rmin**(-self.etamR[f])*np.conjugate(cmr2[-arr_i[f] + Nmax/2])
        cmsymr0[~f] = self.rmin**(-self.etamR[~f])* cmr0[arr_i[~f] - Nmax/2]
        cmsymr2[~f] = self.rmin**(-self.etamR[~f])* cmr2[arr_i[~f] - Nmax/2]

        cmsymr0[-1] = cmsymr0[-1] / 2
        cmsymr0[0] = cmsymr0[0] / 2
        cmsymr2[-1] = cmsymr2[-1] / 2
        cmsymr2[0] = cmsymr2[0] / 2

        P0t = np.real(cmsymr0*self.J0k_arr).sum(axis=1)
        P2t = np.real(cmsymr2*self.J2k_arr).sum(axis=1)

        P0int = interpolate.InterpolatedUnivariateSpline(self.kbins3,P0t)
        P2int = interpolate.InterpolatedUnivariateSpline(self.kbins3,P2t)

        # Now compute chi^2
        chi2 = 0.
        x1 = np.hstack([P0int(self.k)-self.Pk0,P2int(self.k)-self.Pk2])

        chi2 = np.inner(x1,np.inner(self.invcov,x1));
        chi2 = chi2 + (b2 - 0.)**2./1**2. + (bG2 - 0.)**2/1**2. + (css0)**2/30**2 + css2**2/30**2 + (b4-500.)**2/500**2 + (Pshot - 5e3)**2./(5e3)**2.
        loglkl = -0.5 * chi2

        return loglkl
