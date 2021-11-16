# Faraday-Dreams

[![Build Status](https://app.travis-ci.com/Luke-Pratley/Faraday-Dreams.svg?branch=master)](https://app.travis-ci.com/Luke-Pratley/Faraday-Dreams) [![codecov](https://codecov.io/gh/Luke-Pratley/Faraday-Dreams/branch/master/graph/badge.svg?token=V1LGYZVUF0)](https://codecov.io/gh/Luke-Pratley/Faraday-Dreams)


New developments that allow correction for channel averaging when reconstructing broad band polarimetric signals.


This code is built using [Optimus Primal](https://github.com/Luke-Pratley/Optimus-Primal) to apply convex optimization algorithms in deniosing.

## Requirements
- Python 3.8
- [Optimus Primal](https://github.com/Luke-Pratley/Optimus-Primal)
- [PyNUFFT](https://github.com/jyhmiinlin/pynufft) when using the NUFFT algorithm (Optional)

## Install
You can install from the master branch
```
pip install git+https://github.com/Luke-Pratley/Faraday-Dreams.git@master#egg=faradaydreams
```
or from a frozen version at [pypi](https://pypi.org/project/faradaydreams/)
```
pip install faradaydreams
```

## Related Publications
- Removing non-physical structure in fitted Faraday rotated signals: non-parametric QU-fitting, PASA (Accepted), L. Pratley and M. Johnston-Hollitt, and B. M. Gaensler, 2021
- [Wide-band Rotation Measure Synthesis](https://ui.adsabs.harvard.edu/abs/2020ApJ...894...38P/abstract), ApJ, L. Pratley, M. Johnston-Hollitt, 2020

## Basic example for 1d spectrum
We can import a simple solver using the following command
```
import faradaydreams.recover_faraday_spectra as recover_faraday_spectra
```
We can define a dictionary with variables such as the tolerance for convergence `tol`, the maximum number of iterations `iter`, and if we wish to enforce that $\lambda^2 \leq 0$ has zero flux `project_positive_lambda2`. We expect the Faraday depth signal to be complex valued, so the real and positivity constraints are `False`.
```
options = {
    'tol': 1e-5,
    'iter': 50000,
    'update_iter': 1000,
    'record_iters': False,
    'real': False,
    'positivity': False,
    'project_positive_lambda2': False
}
```

Below we pass data to the solver which returns a dictionary with processed result. We pass in the Stokes Q and U spectra and their associated errors in Jy.
We provide the channel frequencies (in Hz) and with `channel_width=None` we do not correct for channel averaging, if we pass the array of channel widths in Hz it will try to do a correction.

The size of a Faraday width pixel is chosen to be `dphi=15` rad/m^2 and we choose the spectrum to span `Nphi=320` pixels across centred at 0 rad/m^2. 
We choose not to scale the uncertainty on the measurements that we wish to fit the solution with `sigma_factor=1`. 
`beta=1e-2` is the step size for the minimization, and `1e-0` to `1e-3` normally works well.
```
results = recover_faraday_spectra.recover_1d_faraday_spectrum(
    Q=data['Q'],
    U=data['U'],
    error_Q=data['e_Q'],
    error_U=data['e_U'],
    freq=freq,
    dphi=15,
    Nphi=320,
    sigma_factor=1,
    beta=1e-2,
    options=options,
    channel_widths=None)     
```
The returned dictionary has the form
```
    return {
        "solution": solution,
        "phi": phi,
        "y_solution": y_model,
        "solution_lambda2": model_lambda2,
        "measurements": y,
        "measurements_residuals": y - m_op.dir_op(solution) / weights,
        "measurements_lambda2": lambda2,
        "measurements_weights": weights,
        "measurements_sigma": sigma
    }
```
where we can plot the solution `results['solution']` against `results['phi']`. `results['y_solution']` is the same signal in lambda^2 coordinates `results['solution_lambda2']`. The original measurements `results['measurements']` and the residuals of the solution `results['measurements_residuals']` are also returned. 

Look in the examples directory for a fully working example in the form of a python notebook.

