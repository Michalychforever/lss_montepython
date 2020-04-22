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

class bao_fs_ngc_z1(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        self.k = np.zeros(self.ksize,'float64')
        self.Pk0 = np.zeros(self.ksize,'float64')
        self.Pk2 = np.zeros(self.ksize,'float64')
        self.alphas = np.zeros(2,'float64')


        self.invcov = np.zeros(
            (2*self.ksize+2, 2*self.ksize+2), 'float64')
        cov = np.zeros(
            (2*self.ksize+2, 2*self.ksize+2), 'float64')


        datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
        for i in range(2*self.ksize+2):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            for j in range(2*self.ksize+2):
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

        datafile = open(os.path.join(self.data_directory, self.alpha_means), 'r')
        for i in range(2):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.alphas[i] = float(line.split()[0])
        datafile.close()


        h = cosmo.h()


        norm = (data.mcmc_parameters['norm']['current'] *
                data.mcmc_parameters['norm']['scale'])
        # norm = 1.
        i_s=repr(3)
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
        Nmax = 128
        b = -1.1001
        kmax = 100.
        k0 = 1.e-4

        rtab = np.zeros(Nmax)
        rmin = 0.01
        rmax = 1000.

        Delta = log(kmax/k0) / (Nmax - 1)
        Delta_r = log(rmax/rmin) / (Nmax - 1)

        Pdiscrin0 = np.zeros(Nmax);
        Pdiscrin2 = np.zeros(Nmax);
        jsNm = np.arange(-Nmax/2,Nmax/2+1,1)
        etam = b + 2*1j*pi*(jsNm)/Nmax/Delta
        kbins3 = np.zeros(Nmax);
        
        z = self.z;
        fz = cosmo.scale_independent_growth_factor_f(z)
        for i in range(Nmax):
    
            kbins3[i] = k0 * exp(Delta * i)
            kinloop1 = kbins3[i] * h
            rtab[i] = rmin * exp(Delta_r * i)

            theory2 = (norm**2.*cosmo.pk(kinloop1, self.z)[18] +norm**4.*(cosmo.pk(kinloop1, self.z)[24])+ norm**1.*b1*cosmo.pk(kinloop1, self.z)[19] +norm**3.*b1*(cosmo.pk(kinloop1, self.z)[25]) + b1**2.*norm**2.*cosmo.pk(kinloop1, self.z)[26] +b1*b2*norm**2.*cosmo.pk(kinloop1, self.z)[34]+ b2*norm**3.*cosmo.pk(kinloop1, self.z)[35] + b1*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[36]+ bG2*norm**3.*cosmo.pk(kinloop1, self.z)[37]  + 2.*(css2 + 0.*b4*kinloop1**2.)*norm**2.*cosmo.pk(kinloop1, self.z)[12]/h**2. + (2.*bG2+0.8*bGamma3)*norm**3.*cosmo.pk(kinloop1, self.z)[9])*h**3. + fz**2.*b4*kbins3[i]**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*cosmo.pk(kinloop1, self.z)[13]*h 
            theory0 = (norm**2.*cosmo.pk(kinloop1, self.z)[15] +norm**4.*(cosmo.pk(kinloop1, self.z)[21])+ norm**1.*b1*cosmo.pk(kinloop1, self.z)[16] +norm**3.*b1*(cosmo.pk(kinloop1, self.z)[22]) + norm**0.*b1**2.*cosmo.pk(kinloop1, self.z)[17] +norm**2.*b1**2.*cosmo.pk(kinloop1, self.z)[23] + 0.25*norm**2.*b2**2.*cosmo.pk(kinloop1, self.z)[1] +b1*b2*norm**2.*cosmo.pk(kinloop1, self.z)[30]+ b2*norm**3.*cosmo.pk(kinloop1, self.z)[31] + b1*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[32]+ bG2*norm**3.*cosmo.pk(kinloop1, self.z)[33] + b2*bG2*norm**2.*cosmo.pk(kinloop1, self.z)[4]+ bG2**2.*norm**2.*cosmo.pk(kinloop1, self.z)[5] + 2.*css0*norm**2.*cosmo.pk(kinloop1, self.z)[11]/h**2. + (2.*bG2+0.8*bGamma3)*norm**2.*(b1*cosmo.pk(kinloop1, self.z)[7]+norm*cosmo.pk(kinloop1, self.z)[8]))*h**3.+Pshot + fz**2.*b4*kbins3[i]**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*cosmo.pk(kinloop1, self.z)[13]*h           

            
#            theory2 = (norm**2.*cosmo.pk(kinloop1, z)[18] +norm**4.*cosmo.pk(kinloop1, z)[24]+ norm**2.*b1*cosmo.pk(kinloop1, z)[19] +norm**4.*b1*cosmo.pk(kinloop1, z)[25] + b1**2.*norm**4.*cosmo.pk(kinloop1, z)[26] +b1*b2*norm**4.*cosmo.pk(kinloop1, z)[34]+ b2*norm**4.*cosmo.pk(kinloop1, z)[35] + b1*bG2*norm**4.*cosmo.pk(kinloop1, z)[36]+ bG2*norm**4.*cosmo.pk(kinloop1, z)[37]  + 2.*c2*norm**2.*cosmo.pk(kinloop1, z)[12]*b1/h**2. + norm**4.*(2.*bG2+0.8*bGamma3)*cosmo.pk(kinloop1, z)[9])*h**3.
#            theory0 = (norm**2.*cosmo.pk(kinloop1, z)[15] +norm**4.*cosmo.pk(kinloop1, z)[21]+ norm**2.*b1*cosmo.pk(kinloop1, z)[16] +norm**4.*b1*cosmo.pk(kinloop1, z)[22]+ norm**2.*b1**2.*cosmo.pk(kinloop1, z)[17] +norm**4.*b1**2.*cosmo.pk(kinloop1, z)[23] + 0.25*norm**4.*b2**2.*cosmo.pk(kinloop1, z)[1] +b1*b2*norm**4.*cosmo.pk(kinloop1, z)[30]+ b2*norm**4.*cosmo.pk(kinloop1, z)[31] + b1*bG2*norm**4.*cosmo.pk(kinloop1, z)[32]+ bG2*norm**4.*cosmo.pk(kinloop1, z)[33]+ b2*bG2*norm**4.*cosmo.pk(kinloop1, z)[4]+ bG2**2.*norm**4.*cosmo.pk(kinloop1, z)[5] + 2.*c0*norm**2.*b1**2.*cosmo.pk(kinloop1, z)[11]/h**2. + norm**4.*(2.*bG2+0.8*bGamma3)*(b1*cosmo.pk(kinloop1, z)[7]+cosmo.pk(kinloop1, z)[8]))*h**3. + Pshot  

            Pdiscrin0[i] = theory0 * exp(-1.*(kinloop1/2.)**4.-1.*b*i*Delta)
            Pdiscrin2[i] = theory2 * exp(-1.*(kinloop1/2.)**4.-1.*b*i*Delta)


        cm0 = np.fft.fft(Pdiscrin0)/ Nmax
        cm2 = np.fft.fft(Pdiscrin2)/ Nmax
        cmsym0 = np.zeros(Nmax+1,dtype=np.complex_)
        cmsym2 = np.zeros(Nmax+1,dtype=np.complex_)

        for i in range(Nmax+1):
            if (i+2 - Nmax/2) < 1:
                cmsym0[i] =  k0**(-etam[i])*np.conjugate(cm0[-i + Nmax/2])
                cmsym2[i] =  k0**(-etam[i])*np.conjugate(cm2[-i + Nmax/2])
            else:
                cmsym0[i] = k0**(-etam[i])* cm0[i - Nmax/2]
                cmsym2[i] = k0**(-etam[i])* cm2[i - Nmax/2]

        cmsym0[-1] = cmsym0[-1] / 2
        cmsym0[0] = cmsym0[0] / 2
        cmsym2[-1] = cmsym2[-1] / 2
        cmsym2[0] = cmsym2[0] / 2

        def J0(r,nu):
            return -1.*np.sin(pi*nu/2.)*r**(-3.-1.*nu)*special.gamma(2+nu)/(2.*pi**2.)
#            return -1j*((-1j*r)**nu - (1j*r)**nu)*r**(-3.-2.*nu)*special.gamma(2+nu)/(4.*pi**2.)
        def J2(r,nu):
            return r**(-3.-1.*nu)*(3.+nu)*special.gamma(2.+nu)*np.sin(pi*nu/2.)/(nu*2.*pi**2.)
        xi0 = np.zeros(Nmax)
        xi2 = np.zeros(Nmax)

        for i in range(Nmax):
            for j in range(Nmax + 1):
                xi0[i] = xi0[i] + np.real(cmsym0[j]*J0(rtab[i],etam[j]))
                xi2[i] = xi2[i] + np.real(cmsym2[j]*J2(rtab[i],etam[j]))

        W0 = np.zeros(Nmax)
        W2 = np.zeros(Nmax)
        W4 = np.zeros(Nmax)
        datafile = open(os.path.join(self.data_directory, self.window_file), 'r')
        for i in range(Nmax):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            W0[i] = float(line.split()[0])
            W2[i] = float(line.split()[1])
            W4[i] = float(line.split()[2]) 
        
        bR = -2.001
        Xidiscrin0 = np.zeros(Nmax);
        Xidiscrin2 = np.zeros(Nmax);
        etamR = bR + 2*1j*pi*(jsNm)/Nmax/Delta_r

        for i in range(Nmax):
            Xidiscrin0[i] = (xi0[i]*W0[i] + 0.2*xi2[i]*W2[i])* exp(-1.*bR*i*Delta_r)
            Xidiscrin2[i] = (xi0[i]*W2[i] + xi2[i]*(W0[i] + 2.*(W2[i]+W4[i])/7.))* exp(-1.*bR*i*Delta_r)

#        fs = 0.6
#        Dfc = 0.33*h
#        def wfc(r,mu):
#            if (np.sqrt(1-mu**2.)*r<Dfc):
#                return 1.
#            else:
#                return 0.

#        for i in range(Nmax):
#            integr0 = lambda mu: (-fs*wfc(rtab[i],mu))*(1. + Xidiscrin0[i] + Xidiscrin2[i]*(-1. + 3.*mu**2.)/2.)
#            integr2 = lambda mu: (-fs*wfc(rtab[i],mu))*(1. + Xidiscrin0[i] + Xidiscrin2[i]*(-1. + 3.*mu**2.)/2.)*(-1. + 3.*mu**2.)/2.
#            Xidiscrin0[i] = Xidiscrin0[i] + integrate.quad(integr0, -1, 1, limit=100)[0]/2
#            Xidiscrin2[i] = Xidiscrin2[i] + integrate.quad(integr2, -1, 1, limit=100)[0]*5/2



        cmr0 = np.fft.fft(Xidiscrin0)/ Nmax
        cmr2 = np.fft.fft(Xidiscrin2)/ Nmax

        cmsymr0 = np.zeros(Nmax+1,dtype=np.complex_)
        cmsymr2 = np.zeros(Nmax+1,dtype=np.complex_)

        for i in range(Nmax+1):
            if (i+2 - Nmax/2) < 1:
                cmsymr0[i] =  rmin**(-etamR[i])*np.conjugate(cmr0[-i + Nmax/2])
                cmsymr2[i] =  rmin**(-etamR[i])*np.conjugate(cmr2[-i + Nmax/2])
            else:
                cmsymr0[i] = rmin**(-etamR[i])* cmr0[i - Nmax/2]
                cmsymr2[i] = rmin**(-etamR[i])* cmr2[i - Nmax/2]

        cmsymr0[-1] = cmsymr0[-1] / 2
        cmsymr0[0] = cmsymr0[0] / 2
        cmsymr2[-1] = cmsymr2[-1] / 2
        cmsymr2[0] = cmsymr2[0] / 2

        def J0k(k,nu):
            return -1.*k**(-3.-1.*nu)*special.gamma(2+nu)*np.sin(pi*nu/2.)*(4.*pi)
        def J2k(k,nu):
            return k**(-3.-1.*nu)*(3.+nu)*special.gamma(2.+nu)*np.sin(pi*nu/2.)*4.*pi/nu

        P0t = np.zeros(Nmax)
        P2t = np.zeros(Nmax)

        for i in range(Nmax):
            for j in range(Nmax + 1):
                P0t[i] = P0t[i] + np.real(cmsymr0[j]*J0k(kbins3[i],etamR[j]))
                P2t[i] = P2t[i] + np.real(cmsymr2[j]*J2k(kbins3[i],etamR[j]))
 
        P0int = interpolate.InterpolatedUnivariateSpline(kbins3,P0t)
        P2int = interpolate.InterpolatedUnivariateSpline(kbins3,P2t)


        rdHfid = 0.040872046001833195;
        rdDAfid = 0.13325653008234437;
        chi2 = 0.
        x1 = np.zeros(2*self.ksize+2)
        for i in range(self.ksize):
            x1[i] = P0int(self.k[i]) - self.Pk0[i]
            x1[self.ksize + i] = P2int(self.k[i]) - self.Pk2[i]
        x1[-2] = rdHfid/(cosmo.rs_drag()*cosmo.Hubble(self.z)) - self.alphas[0]
        x1[-1] = rdDAfid/(cosmo.rs_drag()/cosmo.angular_distance(self.z)) - self.alphas[1]

        chi2 = np.inner(x1,np.inner(self.invcov,x1));
        chi2 = chi2 + (b2 - 0.)**2./1**2. + (bG2 - 0.)**2/1**2. + (css0)**2/30**2 + css2**2/30**2 + (b4-500.)**2/500**2 + (Pshot - 5e3)**2./(5e3)**2.
        loglkl = -0.5 * chi2

        # print("ngcz1 chi2=",chi2)
        return loglkl
