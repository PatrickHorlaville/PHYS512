import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as un

data = np.load('sidebands.npz')
t = data['time']
d = data['signal']

sigma = np.std(d)
sc_sigma = sigma/np.sqrt(len(d))

dt = 10**(-8)
deriv = lambda f, t: 1/(2*dt)*(f(t + dt) - f(t - dt))

def lorentzian_3lor(p, t):
    
    a, b, c, t0, t1, w = p
    y = a/(1 + ((t - t0)/w)**2) + b/(1 + ((t - t0 + t1)/w)**2) + c/(1 + ((t - t0 - t1)/w)**2)

    fa = lambda x1: x1/(1 + ((t - t0)/w)**2) + b/(1 + ((t - t0 + t1)/w)**2) + c/(1 + ((t - t0 - t1)/w)**2)
    fb = lambda x2: a/(1 + ((t - t0)/w)**2) + x2/(1 + ((t - t0 + t1)/w)**2) + c/(1 + ((t - t0 - t1)/w)**2)
    fc = lambda x3: a/(1 + ((t - t0)/w)**2) + b/(1 +((t - t0 + t1)/w)**2) + x3/(1 + ((t - t0 - t1)/w)**2)
    ft0 = lambda x4: a/(1 + ((t - x4)/w)**2) + b/(1 +((t - x4 + t1)/w)**2) + c/(1 + ((t - x4 - t1)/w)**2)
    ft1 = lambda x5: a/(1 + ((t - t0)/w)**2) + b/(1 + ((t - t0 + x5)/w)**2) + c/(1 + ((t - t0 - x5)/w)**2)
    fw = lambda x6: a/(1 + ((t - t0)/x6)**2) + b/(1 + ((t - t0 + t1)/x6)**2) + c/(1 + ((t - t0 - t1)/x6)**2)

    grad = np.zeros([len(t), len(p)])
    grad[:,0] = deriv(fa, a)
    grad[:,1] = deriv(fb, b)
    grad[:,2] = deriv(fc, c)
    grad[:,3] = deriv(ft0, t0)
    grad[:,4] = deriv(ft1, t1)
    grad[:,5] = deriv(fw, w)

    return y, grad

def newtons_method_3lor(data, p, t, n):
    
    y_true = data['signal']
    p_new = p
    
    for i in range(n):
        calc, grad = lorentzian_3lor(p_new, t)  
        res = y_true - calc  
        res = np.matrix(res).T
        grad = np.matrix(grad)
        dp = np.linalg.pinv(grad.T*grad)*(grad.T*res)
 
        for i in range(len(p_new)):
            p_new[i] += dp[i]
            
        p_new = np.asarray(p_new)
        
    return p_new

p0_3lor= np.array([1.5, 0.3, 0.3, 0.00019, 0.00005, 0.00001])
p_new1d = newtons_method_3lor(data, p0_3lor, t, 100) 

cov_d = np.load("mycov_3lor.npy")
errs_d = np.sqrt(np.diag(cov_d))

print('Errors on parameters are:')
print('Error on a is:', errs_d[0])
print('Error on b is:', errs_d[1])
print('Error on c is:', errs_d[2])
print('Error on t0 is:', errs_d[3])
print('Error on t1 is:', errs_d[4])
print('Error on w is:', errs_d[5])


# 1g)


def chisq(p, t, signal, noise):
    y, grad = lorentzian_3lor(p, t)
    chisq = np.sum(((signal - y/noise)**2))
    return chisq



def MCMC(params, cov, nsteps, noise):
    
    chi_0 = chisq(params, t, d, noise)
    
    nparams = len(params)
    chain = np.zeros([nsteps, nparams])
    chi_vec = np.zeros(nsteps)
    
    counts = 0
    
    cholesky_mat = np.linalg.cholesky(cov)

    for i in range(nsteps):

        if i == int(nsteps/4):
            print('25% Complete.')

        if i == int(nsteps/2):
            print('50% Complete.')
        
        if i == int(3*nsteps/4):
            print('75% Complete.')
            
            
        dparams = cholesky_mat@np.random.randn(nparams)

        # Update parameters

        trial_params = params + dparams
        trial_chisq = chisq(trial_params, t, d, noise)
        
        # Compute change in χ2
        dchisq = trial_chisq - chi_0
        
        # Get probability of accepting
        prob_accept = np.exp(-0.5*dchisq)
        accept = np.random.rand(1) < prob_accept
        
        # If accepted, set new parameters and move on to step.
        if accept: 
            counts +=1
            params = trial_params
            chi_0 = trial_chisq
            
        
        chain[i,:] = params
        chi_vec[i] = chi_0
        
    print('Out of', nsteps, 'steps,', counts, 'were accepted (', 100*counts/nsteps, '%)')
    
    return chain, chi_vec

p_g = p_new1d
step_size = np.sqrt(np.diag(cov_d))
steps = 10000
nparams_t = 6

chain_t, chi_vec_t = MCMC(p_g, cov_d, steps, sc_sigma)


# Save data
data_new = np.empty((len(chi_vec_t), nparams_t + 1))
data_new[:,0] = chi_vec_t
for i in range(nparams_t):
    data_new[:, i + 1] = chain_t[:, i]

# np.savetxt('chain_2.txt', data_new)


# print(chi_vec_t)


# print(cov_d)
# nparams_t = len(p_g)
# chol_t = cholesky_decomp(cov_d)
# print('Now computing dpar...')
# dpar = chol_t@np.random.randn(nparams_t)
# print('Finished computing dpar!')
