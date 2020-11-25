from montepython.likelihood_class import Likelihood
import os
import montepython.io_mp as io_mp
import numpy as np
import warnings
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.special import gamma
from scipy.integrate import simps
from scipy.special.orthogonal import p_roots

class boss_full_th_error(Likelihood):
    """"Likelihood for BAO analysis of reconstructed BOSS data. This uses only one data chunk (LOWZ or CMASS and NGC/SGC) with both monopole + quadrupole.
    We analyze using the theoretical error of Baldauf++16, as described in Philcox++20.
    This gives a posterior for the Alcock-Paczynski parameters (alpha_par,alpha_perp).

    These can then be used in the combined bao_fs_{PATCH}_{REDSHIFT} likelihoods."""
    def __init__(self,path,data,command_line):

        Likelihood.__init__(self,path,data,command_line)
        #
	if self.z_type=='high':
		self.z = np.atleast_1d(self.z[1])
	elif self.z_type=='low':
		self.z = np.atleast_1d(self.z[0])
	else:
		raise Exception('z-type must be "low" or "high"')

        self.z = np.asarray(self.z)
        self.n_bin = np.shape(self.z)[0] # number of redshifts

        print('CONFIGURATION')
        print('-------------')
	    print('Field:',self.field)
        print('Window Function:',self.use_window)
        print('Using Quadrupole:',self.use_quadrupole)
        for z_in in range(self.n_bin):
            print('Input data at z = %.2f: %s'%(self.z[z_in],self.file[z_in]))
        print('')
        if self.n_bin>1:
            raise Exception('Each z bin should be analyzed separately.')

        ## Define parameters for Gaussian quadrature
        n_gauss = 30 # number of Gaussian quadrature points
        [self.gauss_mu,self.gauss_w]=p_roots(n_gauss)

        # read data files in each redshift bin
        for index_z in range(self.n_bin):

            # define input arrays
            if index_z==0:
                self.kk = np.array([], 'float64')
            self.PPk0 = np.array([], 'float64')
            self.PPk2 = np.array([], 'float64')

            with open(os.path.join(self.data_directory,self.file[index_z]),'r') as filein:
                for line in filein:
                    if line.strip() and line.find('#')==-1:
                        # excluding hash lines
                        this_line = line.split()
                        # insert into array
                        if index_z==0:
                            self.kk = np.append(self.kk, float(this_line[0])) # central k
                        self.PPk0 = np.append(self.PPk0, float(this_line[1])) # monopole
                        if self.use_quadrupole:
                            self.PPk2 = np.append(self.PPk2,float(this_line[2])) # quadrupole
                        else:
                            self.PPk2 = np.append(self.PPk2,0.) # empty for consistency

                if index_z==0:
                    self.k_size = np.shape(self.kk)[0]
                    self.mu_size = n_gauss
                    self.k = np.zeros((self.k_size,self.n_bin,self.mu_size),'float64')
                    self.Pk0 = np.zeros_like(self.k)
                    self.Pk2 = np.zeros_like(self.k)
                    self.kmax = max(self.kk)

                    # Assign k values to multidimensional array
                    for index_k in xrange(self.k_size):
                        self.k[index_k] = self.kk[index_k]

                # Assign powers to multidimensional power array
                for index_k in xrange(self.k_size):
                    self.Pk0[index_k,index_z] = self.PPk0[index_k]
                    self.Pk2[index_k,index_z] = self.PPk2[index_k]

        # Load in covariance matrix
        self.all_cov = []
        for index_z in range(self.n_bin):
            this_cov = np.loadtxt(os.path.join(self.data_directory,self.cov_file[index_z]))

            if self.use_quadrupole:
                assert len(this_cov)==len(self.kk)*2, 'Need correct size covariance for monopole+quadrupole analysis'
            else:
                if len(this_cov)==len(self.kk):
                    pass
                elif len(this_cov)==len(self.kk)*2:
                    this_cov = this_cov[:len(self.kk),:len(self.kk)]
                else:
                    raise Exception('Need correct size covariance for monopole-only analysis')

            self.all_cov.append(this_cov)


        # Define maximum k
        self.zmax = max(self.z)
        print('Fitting up to k = %.2f'%max(self.kk))

        # Load in smoothed no wiggle power spectrum fit
	if self.z_type=='low':
		sm_lin_file = self.Pk_sm_lin_file[0]
		lin_file = self.Pk_lin_file[0]
	else:
		sm_lin_file = self.Pk_sm_lin_file[1]
		lin_file = self.Pk_lin_file[1]
        k_sm_lin,pk_sm_lin = np.loadtxt(os.path.join(self.data_directory,sm_lin_file))
        # Create a smooth interpolator
        self.pk_sm_lin_interp = interp1d(k_sm_lin,pk_sm_lin)

        # Load in smoothed no wiggle power spectrum fit
        k_lin,pk_lin = np.loadtxt(os.path.join(self.data_directory,lin_file))
        # Create a smooth interpolator
        self.pk_lin_interp = interp1d(k_sm_lin,pk_sm_lin)

        # # Ensure CLASS returns matter power spectrum
        self.need_cosmo_arguments(data, {'output': 'mPk'}) # get P(k)
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 100.}) # ensure we have correct cut-off

        ## CONVOLUTION PARAMETERS
        if self.use_window:
            ## Define input k and r grids
            self.Nmax = 256
            self.bk0 = -1.1001
            self.bk2 = -1.1001
            self.kmax = 30.
            self.k0 = 3.e-5

            self.rmin = 0.01
            self.rmax = 1000.

            self.Delta = np.log(self.kmax/self.k0) / (self.Nmax - 1)
            self.Delta_r = np.log(self.rmax/self.rmin) / (self.Nmax - 1)

            # arrays for power spectrum model
            self.jsNm = np.arange(-self.Nmax/2,self.Nmax/2+1,1)
            self.etam0 = self.bk0 + 2*1j*np.pi*(self.jsNm)/self.Nmax/self.Delta
            self.etam2 = self.bk2 + 2*1j*np.pi*(self.jsNm)/self.Nmax/self.Delta

            # k and r definitions
            self.i_range = np.arange(self.Nmax)
            self.kbins3 = self.k0 * np.exp(self.Delta * self.i_range)
            self.rtab = self.rmin * np.exp(self.Delta_r * self.i_range)

            self.bR = -2.001
            self.etamR = self.bR + 2*1j*np.pi*(self.jsNm)/self.Nmax/self.Delta_r

            # Load in window function data
	    if self.field=='ngc':
		if self.z_type=='low':
			ind=0
		else:
			ind=1
	    elif self.field=='sgc':
		if self.z_type=='low':
			ind=2
		else:
			ind=3
	    else:
		raise Exception('Field type must be "ngc" or "sgc"')
            dx=np.loadtxt(os.path.join(self.data_directory,self.window_file[ind]), skiprows = 1)

            rt=dx[:,0]
            W0t=dx[:,1]
            W2t=dx[:,3]
            W4t=dx[:,5]
            del dx
            self.W0f = InterpolatedUnivariateSpline(rt,W0t)(self.rtab)
            self.W2f = InterpolatedUnivariateSpline(rt,W2t)(self.rtab)
            self.W4f = InterpolatedUnivariateSpline(rt,W4t)(self.rtab)

            # integral constraint
            self.IC  = 0.

        # Define precision matrix using theoretical error
        self.all_prec = []
        Sigma_nl_prior = data.mcmc_parameters['Sigma-nl']['initial'][0]*data.mcmc_parameters['Sigma-nl']['scale']
        bias_prior = data.mcmc_parameters['gal-bias__1']['initial'][0]*data.mcmc_parameters['gal-bias__1']['scale']

        def pk_model_prior(k):
            ## must be in h/Mpc units
            ## Returns linear theory prediction using prior parameters

            k = k.reshape(-1,1)*np.sqrt(1.+np.power(self.gauss_mu.reshape(1,-1),1.)*0.) # this is trivial but used for shape broadcasting

            # Compute the linear power spectrum from CLASS
            # NB: we convert into h/Mpc units here
            pk_lin_th = self.pk_lin_interp(k)

            # compute smoothed linear power spectrum
            P_sm_lin = self.pk_sm_lin_interp(k)

            if self.z[-1]==0.38:
                f_prior = 0.7115090365640122
	    elif self.z[-1]==0.61:
		f_prior = 0.7886358113464698
            else:
                raise Exception('Must recalculate growth factor!')

            # Compute wiggly part
            O_lin = pk_lin_th/P_sm_lin
            Sigma_damping = np.power(Sigma_nl_prior,2.)*(1.+f_prior*np.power(self.gauss_mu,2.)*(2.+f_prior))
            BAO_damping = np.exp(-np.power(k,2.)*Sigma_damping/2.)

            R = 1.-np.exp(-(k*self.Sigma_smooth)**2./2.)
            bias_prefactor = np.power(bias_prior+f_prior*np.power(self.gauss_mu,2.)*R,2.)

            # Define full anisotropic model
            P_k_mu = bias_prefactor*P_sm_lin*(1.+(O_lin-1.)*BAO_damping)

            # Now define multipoles
            leg2 = 0.5*(3.*np.power(self.gauss_mu,2.)-1.) # L2(mu) (faster than Scipy)
            P0_est = np.matmul(P_k_mu,self.gauss_w.reshape(-1,1)).ravel()/2.
            if self.use_quadrupole:
                P2_est = np.matmul(P_k_mu,(leg2*self.gauss_w).reshape(-1,1)).ravel()*5./2.
            else:
                P2_est = 0.

            return P0_est,P2_est

        ## Compute theoretical error envelope and combine with usual covariance
        for index_z in xrange(self.n_bin):

            if self.use_window:
                p0_windowed_int,p2_windowed_int = self.convolve_theory(pk_model_prior,data.cosmo_arguments['h'])
                P0_predictions = p0_windowed_int(self.kk)
                P2_predictions = p2_windowed_int(self.kk)
            else:
                all_predictions = pk_model_prior(self.kk)
                P0_predictions = all_predictions[0]
                P2_predictions = all_predictions[1]

            # Compute the linear (fiducial) power spectrum from CLASS
            envelope_power0 = P0_predictions*2 # extra factor to ensure we don't underestimate error
            envelope_power2 = P0_predictions*np.sqrt(5.) # rescale by sqrt{2ell+1}

            # Define model power
            if self.inflate_error:
                envelope_power0*=5.
                envelope_power2*=5.

            ## COMPUTE THEORETICAL ERROR COVARIANCE
            # Define coupling matrix
            k_mats = np.meshgrid(self.kk,self.kk)
            diff_k = k_mats[0]-k_mats[1]
            rho_submatrix = np.exp(-diff_k**2./(2.*self.Delta_k**2.))

            if self.use_quadrupole:
                # Assume uncorrelated monopole / quadrupole here
                zero_matrix = np.zeros_like(rho_submatrix)
                rho_matrix = np.hstack([np.vstack([rho_submatrix,zero_matrix]),
                                            np.vstack([zero_matrix,rho_submatrix])])
            else:
                rho_matrix = rho_submatrix
            if self.z[-1]==0.38:
                Dz = 0.8195195886296416
                D0 = 1.0
	    elif self.z[-1]==0.61:
		Dz = 0.7299070865701657
		D0 = 1.0
            else:
                raise Exception('Must recalculate fiducial D(z)')
            # Define error envelope from Baldauf'16

            E_vector0 = (Dz/D0)**2.*np.power(self.kk/0.31,1.8)*envelope_power0
            if self.use_quadrupole:
                E_vector2 = (Dz/D0)**2.*np.power(self.kk/0.31,1.8)*envelope_power2
                stacked_E = np.concatenate([E_vector0,E_vector2])
            else:
                stacked_E = E_vector0

            E_mat= np.diag(stacked_E)
            cov_theoretical_error = np.matmul(E_mat,np.matmul(rho_matrix,E_mat))

            # Compute precision matrix
            full_cov = cov_theoretical_error+self.all_cov[index_z]

            full_prec = np.linalg.inv(full_cov)*float(self.N_mocks-2.-self.k_size)/float(self.N_mocks-1.)
            self.all_prec.append(full_prec)

        # end of initialization

    # Define transform functions
    def J0(self,r,nu):
        return -1.*np.sin(np.pi*nu/2.)*r**(-3.-1.*nu)*gamma(2+nu)/(2.*np.pi**2.)
    def J2(self,r,nu):
        return -1.*r**(-3.-1.*nu)*(3.+nu)*gamma(2.+nu)*np.sin(np.pi*nu/2.)/(nu*2.*np.pi**2.)

    # Define inverse transform functions
    def J0k(self,k,nu):
        return -1.*k**(-3.-1.*nu)*gamma(2+nu)*np.sin(np.pi*nu/2.)*(4.*np.pi)
    def J2k(self,k,nu):
        return -1.*k**(-3.-1.*nu)*(3.+nu)*gamma(2.+nu)*np.sin(np.pi*nu/2.)*4.*np.pi/nu

    def convolve_theory(self,p_func,h):
        """Convolve theory with window function
        Inputs are functions giving theory predictions for monopole and quadrupole
        They must take argument as k in h/Mpc units only
        """

        # k bins with correct h value
        kinloop1 = self.kbins3 * h # put in physical units
        p0_prefactor = np.exp( -1.*(kinloop1/2.)**4.-1.*self.bk0*self.i_range*self.Delta)
        if self.use_quadrupole:
            p2_prefactor = np.exp( -1.*(kinloop1/2.)**4. -1.*self.bk2*self.i_range*self.Delta)

        # Compute theoretical power
        p_out = p_func(self.kbins3)
        P0noW = p_out[0]
        if self.use_quadrupole:
            P2noW = p_out[1]
        else:
            P2noW = np.zeros_like(P0noW)

        Pdiscrin0 = P0noW * p0_prefactor
        if self.use_quadrupole:
            Pdiscrin2 = P2noW * p2_prefactor

        # Now FFT power spectra
        cm0 = np.fft.fft(Pdiscrin0)/ self.Nmax
        if self.use_quadrupole:
            cm2 = np.fft.fft(Pdiscrin2)/ self.Nmax

        cmsym0 = np.zeros(self.Nmax+1,dtype=np.complex_)
        if self.use_quadrupole:
            cmsym2 = np.zeros(self.Nmax+1,dtype=np.complex_)

        for i in range(self.Nmax+1):
            if (i+2 - self.Nmax/2) < 1:
                cmsym0[i] =  self.k0**(-self.etam0[i])*np.conjugate(cm0[-i + self.Nmax//2])
                if self.use_quadrupole:
                    cmsym2[i] =  self.k0**(-self.etam2[i])*np.conjugate(cm2[-i + self.Nmax//2])
            else:
                cmsym0[i] = self.k0**(-self.etam0[i])* cm0[i - self.Nmax//2]
                if self.use_quadrupole:
                    cmsym2[i] = self.k0**(-self.etam2[i])* cm2[i - self.Nmax//2]

        cmsym0[-1] = cmsym0[-1] / 2
        cmsym0[0] = cmsym0[0] / 2
        if self.use_quadrupole:
            cmsym2[-1] = cmsym2[-1] / 2
            cmsym2[0] = cmsym2[0] / 2

        # Compute 2PCFs;
        xi0 = np.real(np.matmul(cmsym0,self.J0(self.rtab.reshape(-1,1),self.etam0).T))
        if self.use_quadrupole:
            xi2 = np.real(np.matmul(cmsym2,self.J2(self.rtab.reshape(-1,1),self.etam2).T))
        else:
            xi2 = 0. # this was zero anyhow

        # Convolve multipoles with power spectrum
        Xidiscrin0 = ((xi0 - self.IC)*self.W0f + 0.2*xi2*self.W2f)* np.exp(-1.*self.bR*self.i_range*self.Delta_r)
        if self.use_quadrupole:
            Xidiscrin2 = ((xi0 - self.IC)*self.W2f + xi2*(self.W0f + 2.*(self.W2f+self.W4f)/7.) )* np.exp(-1.*self.bR*self.i_range*self.Delta_r)

        # Do inverse FFTs
        cmr0 = np.fft.fft(Xidiscrin0)/ self.Nmax
        if self.use_quadrupole:
            cmr2 = np.fft.fft(Xidiscrin2)/ self.Nmax

        cmsymr0 = np.zeros(self.Nmax+1,dtype=np.complex_)
        if self.use_quadrupole:
            cmsymr2 = np.zeros(self.Nmax+1,dtype=np.complex_)

        for i in range(self.Nmax+1):
            if (i+2 - self.Nmax/2) < 1:
                cmsymr0[i] =  self.rmin**(-self.etamR[i])*np.conjugate(cmr0[-i + self.Nmax//2])
                if self.use_quadrupole:
                    cmsymr2[i] =  self.rmin**(-self.etamR[i])*np.conjugate(cmr2[-i + self.Nmax//2])
            else:
                cmsymr0[i] = self.rmin**(-self.etamR[i])* cmr0[i - self.Nmax//2]
                if self.use_quadrupole:
                    cmsymr2[i] = self.rmin**(-self.etamR[i])* cmr2[i - self.Nmax//2]

        cmsymr0[-1] = cmsymr0[-1] / 2
        cmsymr0[0] = cmsymr0[0] / 2
        if self.use_quadrupole:
            cmsymr2[-1] = cmsymr2[-1] / 2
            cmsymr2[0] = cmsymr2[0] / 2

        # Compute windowed predictions
        P0t = np.real(np.matmul(cmsymr0,self.J0k(self.kbins3.reshape(-1,1),self.etamR).T))
        if self.use_quadrupole:
            P2t = np.real(np.matmul(cmsymr2,self.J2k(self.kbins3.reshape(-1,1),self.etamR).T))

        # Create interpolator for P0,P2
        P0int = InterpolatedUnivariateSpline(self.kbins3,P0t)
        if self.use_quadrupole:
            P2int = InterpolatedUnivariateSpline(self.kbins3,P2t)
        else:
            P2int = lambda k: 0. # empty function for consistency

        return P0int,P2int

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.0

        # get AP rescaling alpha and rescale momentum
        alpha_par = data.mcmc_parameters['alpha-par']['current']*data.mcmc_parameters['alpha-par']['scale']
        alpha_perp = data.mcmc_parameters['alpha-perp']['current']*data.mcmc_parameters['alpha-perp']['scale']
        Sigma_nl = data.mcmc_parameters['Sigma-nl']['current']*data.mcmc_parameters['Sigma-nl']['scale']

        # Get bias vector from parameters
        bias1 = data.mcmc_parameters['gal-bias__1']['current']*data.mcmc_parameters['gal-bias__1']['scale']
        all_bias = [bias1]

        def pk_model(k):
            ## must be in h/Mpc units
            ## Returns linear theory prediction for monopole and quadrupole

            ## Compute AP-rescaled parameters
            F = alpha_par/alpha_perp

            k1 = k.reshape(-1,1)/alpha_perp*np.sqrt(1.+np.power(self.gauss_mu,2.)*(np.power(F,-2.)-1.))
            k1 = k1[:,np.newaxis,:] # reshape for later use
            mu1 = self.gauss_mu/(F*np.sqrt(1.+np.power(self.gauss_mu,2.)*(np.power(F,-2.)-1.)))

            # Compute the linear power spectrum from CLASS at the k1 positions
            # NB: we convert into h/Mpc units here

            pk_lin_th = np.power(cosmo.h(),3.)*cosmo.get_pk_lin(k1*cosmo.h(),self.z,len(k),self.n_bin, self.mu_size)

            # compute smoothed linear power spectrum
            P_sm_lin = self.pk_sm_lin_interp(k1)

            f = cosmo.scale_independent_growth_factor_f(self.z)

            # Compute wiggly part
            O_lin = pk_lin_th/P_sm_lin
            Sigma_damping = np.power(Sigma_nl,2.)*(1.+f*np.power(mu1,2.)*(2.+f))
            BAO_damping = np.exp(-np.power(k1,2.)*Sigma_damping/2.)

            # Compute bias prefactor
            R = 1.-np.exp(-(k1*self.Sigma_smooth)**2./2.)
            bias_prefactor = np.power(bias1+f*np.power(mu1,2.)*R,2.)

            # Define full anisotropic model
            P_k_mu = bias_prefactor*P_sm_lin*(1.+(O_lin-1.)*BAO_damping)

            # Now define multipoles
            leg2 = 0.5*(3.*np.power(self.gauss_mu,2.)-1.) # L2(mu) (faster than Scipy)

            # Use Gaussian quadrature for fast integral evaluation
            P0_est = np.matmul(P_k_mu,self.gauss_w.reshape(-1,1)).ravel()/2.

            if self.use_quadrupole:
                P2_est = np.matmul(P_k_mu,(leg2*self.gauss_w).reshape(-1,1)).ravel()*5./2.
            else:
                P2_est = 0.

            return P0_est,P2_est

        ## Compute windowed power spectrum multipoles
        if self.use_window:
            p0_windowed_int,p2_windowed_int = self.convolve_theory(pk_model,cosmo.h())
            P0_predictions = p0_windowed_int(self.kk)
            P2_predictions = p2_windowed_int(self.kk)
        else:
            P0_predictions = pk0_model(self.kk)
            P2_predictions = pk2_model(self.kk)

        # Compute chi2 for each z-mean
        for index_z in xrange(self.n_bin):

            # Load in full precision matrix with theoretical error
            full_prec = self.all_prec[index_z]

            # Create vector of residual pk
            if self.use_quadrupole:
                stacked_model = np.concatenate([P0_predictions,P2_predictions])
                stacked_data = np.concatenate([self.Pk0[:,index_z,0],self.Pk2[:,index_z,0]])
            else:
                stacked_model = P0_predictions
                stacked_data = self.Pk0[:,index_z,0]
            resid_vec = (stacked_data-stacked_model).reshape(-1,1)

            # NB: should use cholesky decomposition and triangular factorization when we need to invert arrays later
	        mb = 0 # minimum bin
            chi2+=float(np.matmul(resid_vec[mb:].T,np.matmul(full_prec[mb:,mb:],resid_vec[mb:])))

        lkl =  -0.5*chi2
        return lkl
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
