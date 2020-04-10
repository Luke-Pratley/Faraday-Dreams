import optimusprimal.prox_operators as prox_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np


def constrined_solver(data, sigma, phi, psi, beta = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}):
    """
    Solve constrained l1 regularization problem
    """

    size = len(data)
    epsilon = np.sqrt(size + 2. * np.sqrt(size)) * sigma
    p = prox_operators.l2_ball(epsilon, data, phi)
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(data))) * beta, psi)
    return primal_dual.FBPD(phi.adj_op(data), options, None, h, p, None)


def unconstrined_solver(data, sigma, phi, psi, beta = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}):
    """
    Solve unconstrained l1 regularization problem
    """

    g = grad_operators.l2_norm(sigma, data, phi)
    h = prox_operators.l1_norm(beta, psi)
    return primal_dual.FBPD(phi.adj_op(data), options, None, h, None, g)
