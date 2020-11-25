import os
import numpy as np
from montepython.likelihood_class import Likelihood_prior
from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from scipy.special import gamma,erf
from scipy import interpolate
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special

class ngc_z3_marg(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.

    def __init__(self,path,data,command_line):

        Likelihood_prior.__init__(self,path,data,command_line)

        # First load in data

        self.k = np.zeros(self.ksize,'float64')
        self.Pk0 = np.zeros(self.ksize,'float64')
        self.Pk2 = np.zeros(self.ksize,'float64')

        self.cov = np.zeros(
            (2*self.ksize, 2*self.ksize), 'float64')

        datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
        for i in range(2*self.ksize):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            for j in range(2*self.ksize):
                self.cov[i,j] = float(line.split()[j])
        datafile.close()
        self.logdetcov = np.linalg.slogdet(self.cov)[1]

        datafile = open(os.path.join(self.data_directory, self.measurements_file), 'r')
        for i in range(self.ksize):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.k[i] = float(line.split()[0])
            self.Pk0[i] = float(line.split()[1])
            self.Pk2[i] = float(line.split()[2])

        datafile.close()
        self.stack_dat = np.hstack([self.Pk0,self.Pk2])

        self.Nmax=128
        self.W0 = np.zeros((self.Nmax))
        self.W2 = np.zeros((self.Nmax))
        self.W4 = np.zeros((self.Nmax))
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

        jsNm = np.arange(-self.Nmax//2,self.Nmax//2+1,1)
        self.etam = b + 2*1j*pi*(jsNm)/self.Nmax/Delta

        def J_func(r,nu):
            gam = special.gamma(2+nu)
            r_pow = r**(-3.-1.*nu)
            sin_nu = np.sin(pi*nu/2.)
            J0 = -1.*sin_nu*r_pow*gam/(2.*pi**2.)
            J2 = -1.*r_pow*(3.+nu)*gam*sin_nu/(nu*2.*pi**2.)
            return J0,J2

        self.J0_arr,self.J2_arr = J_func(rtab.reshape(-1,1),self.etam.reshape(1,-1))

        self.etamR = bR + 2*1j*pi*(jsNm)/self.Nmax/Delta_r

        def Jk_func(k,nu):
            gam = special.gamma(2+nu)
            k_pow = k**(-3.-1.*nu)
            sin_nu = np.sin(pi*nu/2.)
            J0k = -1.*k_pow*gam*sin_nu*(4.*pi)
            J2k = -1.*k_pow*(3.+nu)*gam*sin_nu*4.*pi/nu
            return J0k,J2k

        self.J0k_arr,self.J2k_arr = Jk_func(self.kbins3.reshape(-1,1),self.etamR.reshape(1,-1))

        # Compute window response matrix
        resp00 = np.zeros((self.ksize,self.Nmax))
        resp02 = np.zeros((self.ksize,self.Nmax))
        resp20 = np.zeros((self.ksize,self.Nmax))
        resp22 = np.zeros((self.ksize,self.Nmax))
        for i in range(self.Nmax):
            tmp_resp0 = self.window_response(0,i)
            tmp_resp2 = self.window_response(2,i)
            resp00[:,i] = tmp_resp0[0]
            resp20[:,i] = tmp_resp0[1]
            resp02[:,i] = tmp_resp2[0]
            resp22[:,i] = tmp_resp2[1]
        resp0 = np.hstack([resp00,resp02])
        resp2 = np.hstack([resp20,resp22])
        self.response_matrix = np.vstack([resp0,resp2])

    def window_response(self, l_i,k_index):

        Nmax = self.Nmax
        k0 = self.k0

        Pdiscrin0 = np.zeros(Nmax)
        Pdiscrin2 = np.zeros(Nmax)

        if l_i==0:
            Pdiscrin0[k_index] = 1
        if l_i==2:
            Pdiscrin2[k_index] = 1

        cm0 = np.fft.fft(Pdiscrin0)/ Nmax
        cm2 = np.fft.fft(Pdiscrin2)/ Nmax
        cmsym0 = np.zeros(Nmax+1,dtype=np.complex_)
        cmsym2 = np.zeros(Nmax+1,dtype=np.complex_)

        all_i = np.arange(Nmax+1)
        f = (all_i+2-Nmax//2) < 1
        cmsym0[f] = k0**(-self.etam[f])*np.conjugate(cm0[-all_i[f]+Nmax//2])
        cmsym2[f] = k0**(-self.etam[f])*np.conjugate(cm2[-all_i[f]+Nmax//2])
        cmsym0[~f] = k0**(-self.etam[~f])*cm0[all_i[~f]-Nmax//2]
        cmsym2[~f] = k0**(-self.etam[~f])*cm2[all_i[~f]-Nmax//2]

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
        f = (arr_i+2-Nmax//2)<1

        cmsymr0[f] = self.rmin**(-self.etamR[f])*np.conjugate(cmr0[-arr_i[f] + Nmax//2])
        cmsymr2[f] =  self.rmin**(-self.etamR[f])*np.conjugate(cmr2[-arr_i[f] + Nmax//2])
        cmsymr0[~f] = self.rmin**(-self.etamR[~f])* cmr0[arr_i[~f] - Nmax//2]
        cmsymr2[~f] = self.rmin**(-self.etamR[~f])* cmr2[arr_i[~f] - Nmax//2]

        cmsymr0[-1] = cmsymr0[-1] / 2
        cmsymr0[0] = cmsymr0[0] / 2
        cmsymr2[-1] = cmsymr2[-1] / 2
        cmsymr2[0] = cmsymr2[0] / 2

        P0t = np.real(cmsymr0*self.J0k_arr).sum(axis=1)
        P2t = np.real(cmsymr2*self.J2k_arr).sum(axis=1)

        P0int = interpolate.InterpolatedUnivariateSpline(self.kbins3,P0t)(self.k)
        P2int = interpolate.InterpolatedUnivariateSpline(self.kbins3,P2t)(self.k)

        return P0int,P2int

    def loglkl(self, cosmo, data):

        h = cosmo.h()

        norm = (data.mcmc_parameters['norm']['current'] *
                 data.mcmc_parameters['norm']['scale'])
        i_s=repr(1)
        b1 = (data.mcmc_parameters['b^{('+i_s+')}_1']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_1']['scale'])
        b2 = (data.mcmc_parameters['b^{('+i_s+')}_2']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_2']['scale'])
        bG2 = (data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['scale'])

        bGamma3 = 0.
        a2 = 0.
        Nmax = self.Nmax
        k0 = self.k0

        z = self.z;
        fz = cosmo.scale_independent_growth_factor_f(z)

        assert bGamma3 == 0., 'bGamma3 has been set to zero in the derivatives'

        css0sig = 30.
        css2sig = 30.
        b4sig = 500.
        Pshotsig = 5e3
        css0mean = 0.
        css2mean = 30.
        b4mean = 500.
        Pshotmean = 0.
        Nmarg = 4 # number of parameters to marginalize

        # Run CLASS-PT
        all_theory = cosmo.get_pk_mult(self.kbins3*h,self.z, Nmax)

        # Compute usual theory model
        kinloop1 = self.kbins3 * h

        theory2 = (norm**2.*all_theory[18] +norm**4.*(all_theory[24])+ norm**1.*b1*all_theory[19] +norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37]  + 2.*(css2mean + 0.*b4mean*kinloop1**2.)*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3)*norm**3.*all_theory[9])*h**3. + fz**2.*b4mean*self.kbins3**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h

        theory0 = (norm**2.*all_theory[15] +norm**4.*(all_theory[21])+ norm**1.*b1*all_theory[16] +norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*css0mean*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3.+Pshotmean + fz**2.*b4mean*self.kbins3**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h

        # Pieces with linear dependencies on biases
        dtheory2_dcss0 = np.zeros_like(self.kbins3)
        dtheory2_dcss2 = (2.*norm**2.*all_theory[12]/h**2.)*h**3.
        dtheory2_db4 = (2.*(0.*kinloop1**2.)*norm**2.*all_theory[12]/h**2.)*h**3. + fz**2.*self.kbins3**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h
        dtheory2_dPshot = np.zeros_like(self.kbins3)

        dtheory0_dcss0 = (2.*norm**2.*all_theory[11]/h**2.)*h**3.
        dtheory0_dcss2 = np.zeros_like(self.kbins3)
        dtheory0_db4 = fz**2.*self.kbins3**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h
        dtheory0_dPshot = np.ones_like(self.kbins3)

        # Put all into a vector for simplicity
        theory0vec = np.vstack([theory0,dtheory0_dcss0,dtheory0_dcss2,dtheory0_db4,dtheory0_dPshot]).T
        theory2vec = np.vstack([theory2,dtheory2_dcss0,dtheory2_dcss2,dtheory2_db4,dtheory2_dPshot]).T

        i_arr = np.arange(Nmax)
        factor = (exp(-1.*(self.kbins3*h/2.)**4.)*self.tmp_factor)[:,np.newaxis]
        Pdisc = np.vstack([theory0vec*factor,theory2vec*factor])

        # Compute marginalized covariance with window-convolved derivatives
        deriv_stack = np.inner(self.response_matrix,Pdisc[:,1:].T)
        marg_cov = self.cov + css0sig**2*np.outer(deriv_stack[:,0],deriv_stack[:,0]) + css2sig**2*np.outer(deriv_stack[:,1],deriv_stack[:,1]) + b4sig**2*np.outer(deriv_stack[:,2], deriv_stack[:,2]) + Pshotsig**2*np.outer(deriv_stack[:,3],deriv_stack[:,3])

        invcov_marg = np.linalg.inv(marg_cov)

        # Now convolve with window function
        invcovW = np.matmul(self.response_matrix.T,invcov_marg)
        invcovWW = np.matmul(self.response_matrix.T,np.matmul(invcov_marg,self.response_matrix))

        # Now compute chi^2

        chi2 = np.inner(Pdisc[:,0],np.inner(invcovWW,Pdisc[:,0]))
        chi2 += np.inner(self.stack_dat,np.inner(invcov_marg,self.stack_dat))
        chi2 += -2.*np.inner(Pdisc[:,0],np.inner(invcovW,self.stack_dat))

        # Add priors
        chi2 += (b2 - 0.)**2./1**2. + (bG2 - 0.)**2/1**2. #+ (css0)**2/30**2 + css2**2/30**2 + (b4-500.)**2/500**2 + (Pshot - 5e3)**2./(5e3)**2.

        # Add determinants (depending on cosmology)
        chi2 += np.linalg.slogdet(marg_cov)[1] - self.logdetcov # add on trace-log part and remove unmarginalized part (independent of cosmology)

        loglkl = -0.5 * chi2

        return loglkl
