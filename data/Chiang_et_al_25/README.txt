This package contains the data vector, covariance matrix, best-fit model, and summary of astrophysical information from the paper "Cosmic Infrared Background Tomography and a Census of Cosmic Dust and Star Formation" by Chiang, Makiya, and Ménard (2025; arXiv:2504.05384).


(1) Tomographic CIB measurements are presented in two equivalent forms: one in the observer frame as the bias-weighted redshift distribution (dI/dz)b [MJy/sr], and the other in the rest frame as the bias-weighted comoving emissivity (epsilon_nu)b [erg/s/Hz/Mpc^3].

Data vectors for (dI/dz)b and (epsilon_nu)b: "CIB_dIdz_b.ecsv" and "CIB_emissivity_b.ecsv"

Covariance matrices for (dI/dz)b and (epsilon_nu)b: "CIB_dIdz_b_covariance_matrix.txt" and "CIB_emissivity_b_covariance_matrix.txt"

Best-fit models for (dI/dz)b and the bias-corrected dI/dz: "CIB_dIdz_b--best_fit_model.ecsv" and "CIB_dIdz--best_fit_model.ecsv"

Best-fit models for (epsilon_nu)b and the bias-corrected epsilon_nu: "CIB_emissivity_b--best_fit_model.ecsv" and "CIB_emissivity--best_fit_model.ecsv"

(2) The redshift-integrated CIB monopole intensity [MJy/sr] in the observer frame from this work, with and without the thermal SZ effect correction (from Chiang et al. 2020, ApJ 902, 56), is provided in "CIB_tSZ_monopole.csv".

(3) A summary of astrophysical quantities, including dust temperature, dust mass, CIB infrared luminosity density, cosmic star formation rate, and dust-obscured fraction, is provided in "Astrophysics_Summary.ecsv".

Version 2 note: Results in this version are updated from version 1 using the correct Herschel filter response functions for extended sources from the SPIRE Observers’ Manual. Version 1 used those from SVO, which were found to be incorrect.