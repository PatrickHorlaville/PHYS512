import numpy as np
import matplotlib.pyplot as plt
import camb

import time


def get_spectrum(pars, npts, lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2: npts + 2]


params_test = np.array([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
nparams = len(params_test)


def get_deriv(pars, npts):
    
    deriv_mat = np.empty([npts, npars])
    
    for i in range(npars):
        
        h_arr = np.zeros(npars)
        h_arr[i] = 0.01*pars[i]
        
        deriv_mat[:, i] = (get_spectrum(pars + h_arr, npts) - get_spectrum(pars - h_arr, npts))/(2*h_arr[i])
    
    return deriv_mat

def update_lam(lam, yes):
    
    if yes: 
        lam = lam/1.5
        if lam < 0.5: 
            lam = 0
    else: 
        if lam == 0: 
            lam = 1
        else:
            lam = lam*1.5**2
            
    return lam

def get_mat(params, npts, N_inv):
    
    model = get_spectrum(params, npts)
    derivs = get_deriv(params, npts)
    
    res = spec - model
    
    lhs = derivs.T@N_inv@derivs
    rhs = derivs.T@N_inv@res
    chisq = res.T@N_inv@res
    
    return chisq, lhs, rhs

def linv(mat, lam):
    mat = mat + lam*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

def LM_fit(params, npts, N_inv, chi_tol, max_iter, params_prior = None, params_err = None):
    
    lam = 0
    chisq, lhs, rhs = get_mat(params, npts, N_inv)
    
    for i in range(max_iter):
        
        while True: 

            lhs_inv = linv(lhs, lam)

            dparams = lhs_inv@rhs
            
            if (params[3] + dparams[3] > 0.02) and (params[3] + dparams[3] < 0.1):
                break
            
            else:
                lam = update_lam(lam, False)

        
        chisq_new, lhs_new, rhs_new = get_mat(params + dparams, npts, N_inv)
        
        if params_prior is not None:
            trial_prior = prior_chisq(params + dparams, params_prior, params_err)
            chisq_new += trial_prior
        
        
        if chisq_new < chisq:
            
            if lam == 0.0:
                
                if (np.abs(chisq - chisq_new) < chi_tol):
                    return params + dparams, lhs_new
            
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            params = params + dparams
            lam = update_lam(lam, True)
        
        else: 
            lam = update_lam(lam, False)
        
    return params, lhs

best_fit_params_new, cov_inv_new = LM_fit(params = params_test, npts = len(spec), N_inv = Ninv, chi_tol = 0.1, max_iter = 100) 
cov_new = np.linalg.inv(cov_inv_new)
err_params_new = np.sqrt(np.diag(cov_new))
np.savetxt('Results_lot/curvature_matrix_bestfit_lot.txt', cov_new)
np.savetxt('Results_lot/planck_fit_params_lot.txt', np.vstack((best_fit_params_new, err_params_new)))

def get_chisq(params, noise = errs):
    
    model = get_spectrum(params, npts)
    res = spec - model
    chisq = np.sum((res/noise)**2)
    
    return chisq

def MCMC(params, curv, nsteps, noise = errs, params_prior = None, params_err = None):
    
    chi_cur = get_chisq(params, noise)
    
    chain = np.zeros([nsteps, nparams])
    chi_vec = np.zeros(nsteps)
    
    counts = 0
    
    cholesky_mat = np.linalg.cholesky(curv)
    
    for i in range(nsteps):
            
        while True: 
            
            dparams = cholesky_mat@np.random.randn(nparams)

            if (params[3] + dparams[3] > 0.02) and (params[3] + dparams[3] < 0.1):
                break
        
        trial_params = params + dparams
        trial_chisq = get_chisq(trial_params, noise)
        
        if params_prior is not None: 
            trial_chisq += prior_chisq(trial_params, params_prior, params_err)
            
        dchisq = trial_chisq - chi_cur
        
        prob_accept = np.exp(-0.5*dchisq)
        accept = np.random.rand(1) < prob_accept
        
        if accept: 
            counts +=1
            params = trial_params
            chi_cur = trial_chisq
            
        
        chain[i,:] = params
        chi_vec[i] = chi_cur
            
    return chain, chi_vec 
    
chain_new, chisq_new = MCMC(params_test, cov_new, 1000)

data_new = np.empty((len(chisq_new), nparams + 1))
data_new[:,0] = chisq_new
for i in range(nparams):
    data_new[:, i + 1] = chain_new[:, i]

np.savetxt('Results_lot/planck_chain_lot.txt', data_new)

def prior_chisq(params, params_prior, params_err):
    params_shifted = params - params_prior 
    return np.sum((params_shifted/params_err)**2)

params_priors = np.array([0.0, 0.0, 0.0, 0.0540, 0.0, 0.0])
params_errs = np.zeros(nparams) + 1e20
params_errs[3] = 0.0074

best_fit_params_prior_new, cov_inv_prior_new = LM_fit(params = params_test, npts = len(spec), N_inv = Ninv, chi_tol = 0.1, max_iter = 100, params_prior = params_priors, params_err = params_errs) 
cov_prior_new = np.linalg.inv(cov_inv_prior_new)
np.savetxt('Results_lot/curvature_matrix_bestfit_tauprior_lot.txt', cov_prior_new)

chain_prior_new, chisq_prior_new = MCMC(params = params_test, curv = cov_prior_new, nsteps = 1000, params_prior = params_priors, params_err = params_errs)


data_prior_new = np.empty((len(chisq_prior_new), nparams + 1))
data_prior_new[:,0] = chisq_prior_new
for i in range(nparams):
    data_prior_new[:, i + 1] = chain_prior_new[:, i]

np.savetxt('Results_lot/planck_chain_tauprior_lot.txt', data_prior_new)


