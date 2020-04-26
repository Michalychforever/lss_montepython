# lss-montepython

**Contributors:** Mikhail (Misha) Ivanov, Anton Chudaykin, Oliver Philcox 

These are various large-scale structure likelihoods for the MCMC sampler [Montepython](https://github.com/brinckmann/montepython_public) that were analyzed in in the papers 

* [*Measuring neutrino masses with large-scale structure: Euclid forecast with controlled theoretical error*](https://arxiv.org/abs/1907.06666)
* [*Cosmological Parameters from the BOSS Galaxy Power Spectrum*](https://arxiv.org/abs/1909.05277)
* [*Cosmological Parameters and Neutrino Masses from the Final Planck and Full-Shape BOSS Data*](https://arxiv.org/abs/1912.08208) 
* [*Combining Full-Shape and BAO Analyses of Galaxy Power Spectra: A 1.6% CMB-independent constraint on H0*](https://arxiv.org/abs/2002.04035)

The repo includes: 

* Mock power spectrum and bispectrum likelihoods for a Euclid-like survey spanning over eight redshift bins from z=0.6 to z=2 with one-loop or two-loop theoretical error covariances
* Custom-built [BOSS DR12](https://arxiv.org/abs/1607.03155) pre-reconstructed full-shape (FS) power spectrum likelihods for four independent data chunks: North and South Galactic Caps (NGC and SGC) at z=0.38 and z=0.61
* BAO-only likelihood that extracts the anisotropic BAO signal from the post-reconstructed power spectra
* Joint FS+BAO likelihoods with the appropriate covariance matrices for the same data chunks

Note that you need [CLASS-PT](https://github.com/Michalychforever/CLASS-PT) to evaluate FS and FS+BAO likelihoods
