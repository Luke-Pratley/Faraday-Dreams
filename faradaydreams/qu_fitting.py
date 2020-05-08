import emcee
from scipy.optimize import minimize
import numpy as np


def log_likelihood(params, model, channels, y, sigma):
    return -0.5 * np.sum((y - model(channels, params)) ** 2 / sigma**2)


def mcmc_sample(channels, y, sigma, start_params, log_probability, nwalkers, iters):
    ndim = len(start_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(channels, y, sigma))
    sampler.run_mcmc(start_params + 1e-4 * np.random.randn(nwalkers, ndim), iters, progress=True)


def least_squares(params_estimate, model, channels, y, sigma):
    nll = lambda *args: -log_likelihood(*args)
    initial = params_estimate
    return minimize(nll, initial, args=(channels, y, sigma))
