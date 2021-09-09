import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np
from enum import Enum

import logging


class algorithm(Enum):
    l1_constrained = 0
    l1_unconstrained = 1
    l2_unconstrained = 2


logger = logging.getLogger('Faraday Dreams')


def solver(algo, measurements, sigma, phi, wav=["dirac, db1, db2, db3, db4"], levels=6, operator_norm=1, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, "positivity": False, 'real': False, 'project_positive_lambda2': True}, viewer=None, estimate=None):
    logger.info("Using wavelets %s with %s levels", wav, levels)
    logger.info(
        "Using an estimated noise level of %s (weighted image units, i.e. Jy/Beam)", sigma)
    if estimate is None:
        estimate = phi.adj_op(measurements)/operator_norm
    psi = linear_operators.dictionary(wav, levels, estimate.shape)
    if algo == algorithm.l1_constrained:
        logger.info(
            "Reconstructing Faraday Depth using constrained l1 regularization")
        return l1_constrained_solver(estimate, measurements, sigma, phi, psi, operator_norm, beta, options, viewer)
    if algo == algorithm.l1_unconstrained:
        logger.info(
            "Reconstructing Faraday Depth using unconstrained l1 regularization")
        return l1_unconstrained_solver(estimate, measurements, sigma, phi, psi, operator_norm, beta, options, viewer)
    raise ValueError("Algorithm not reconginized.")


def l1_constrained_solver(estimate, measurements, sigma, phi, psi, operator_norm=1, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False, 'real': False, 'project_positive_lambda2': True}, viewer=None):
    """
    Solve constrained l1 regularization problem
    """
    size = len(np.ravel(measurements))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
    p = prox_operators.l2_ball(epsilon, measurements, phi)
    p.beta = operator_norm
    h = prox_operators.l1_norm(
        np.max(np.abs(psi.dir_op(estimate))) * beta, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.real_prox()
    r = None
    if options['project_positive_lambda2'] == True:
        negative_l = np.arange(1, int(estimate.shape[0] * 1./2.))
        mask = np.zeros(estimate.shape, dtype=bool)
        mask[negative_l, ...] = True
        def fft_dir(x):
            return np.fft.fft(x,axis=0)
        def fft_adj(x):
            return np.fft.ifft(x,axis=0)
        r = prox_operators.zero_prox(mask, linear_operators.function_wrapper(fft_dir, fft_adj))
    return primal_dual.FBPD(estimate, options, None, f, h, p, r, viewer)


def l1_unconstrained_solver(estimate, measurements, sigma, phi, psi, operator_norm=1, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}, viewer=None):
    """
    Solve unconstrained l1 regularization problem
    """

    g = grad_operators.l2_norm(sigma, measurements, phi)
    g.beta = operator_norm / sigma**2
    if beta <= 0:
        h = None
    else:
        h = prox_operators.l1_norm(beta, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.real_prox()
    return primal_dual.FBPD(estimate, options, g, f, h, None, viewer)

