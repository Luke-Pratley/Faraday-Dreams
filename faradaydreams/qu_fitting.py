import emcee
from scipy.optimize import minimize
import dynesty
import numpy as np


def loglike(params, model, lambda2, y, sigma):
    return -0.5 * np.sum(np.abs(y - model(lambda2, *params)) ** 2 / sigma**2)


def mcmc_sample(model, lambda2, y, sigma, start_params, nwalkers, iters):
    ndim = len(start_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, args=(model, lambda2, y, sigma))
    sampler.run_mcmc(start_params + 100 * np.random.randn(nwalkers, ndim), iters, progress=True)
    return sampler


def least_squares(params_estimate, model, lambda2, y, sigma):
    nll = lambda *args: -loglike(*args)
    initial = params_estimate
    return minimize(nll, initial, args=(model, lambda2, y, sigma, None), method='BFGS', tol=1e-6, options={'maxiter': 1e5, 'disp':True})

