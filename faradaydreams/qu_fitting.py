import emcee
from scipy.optimize import minimize
import numpy as np


def log_probability(params, model, lambda2, y, sigma, constraint):
    prior = 0.
    return -0.5 * np.sum(np.abs(y - model(lambda2, *params)) ** 2 / sigma**2) + prior


def log_prior(params, constraint):
    if constraint(*params):
        return -np.infty
    else:
        return 0


def mcmc_sample(model, lambda2, y, sigma, start_params, constraint, nwalkers, iters):
    ndim = len(start_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(model, lambda2, y, sigma, constraint))
    sampler.run_mcmc(start_params + 100 * np.random.randn(nwalkers, ndim), iters, progress=True)
    return sampler


def least_squares(params_estimate, model, lambda2, y, sigma):
    nll = lambda *args: -log_probability(*args)
    initial = params_estimate
    return minimize(nll, initial, args=(model, lambda2, y, sigma, None), method='BFGS', tol=1e-6, options={'maxiter': 1e5, 'disp':True})
