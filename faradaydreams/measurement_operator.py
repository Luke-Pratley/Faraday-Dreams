import numpy as np

import logging


logger = logging.getLogger('Faraday Dreams')


def create_faraday_matrix(lambda2, phi, lambda2_width, weights):
    A = np.zeros((len(lambda2), len(phi)), dtype=np.complex)
    for r in range(len(lambda2)):
        A[r, :] = np.exp(2 * lambda2[r] * phi * 1j) * \
            np.sinc(phi * lambda2_width[r]/np.pi) * weights[r]
    return A


def create_delay_matrix(nu, tau, weights):
    A = np.zeros((len(nu), len(tau)), dtype=np.complex)
    for r in range(len(nu)):
        A[r, :] = np.exp(2 * np.pi * nu[r] * tau * 1j) * weights[r]
    return A


class faraday_operator:
    """
Operator that simulates a Faraday rotation observation and finite channel width.

INPUT:
    lambda2 - channel locations in m^2
    phi - pixel locations in rad/m^2
    weights - weights for each channel in m^2
    lambda2_width - channel widths in m^2

"""

    def __init__(self, lambda2, phi, weights, lambda2_width=None):
        if(np.all(lambda2_width != None)):
            assert lambda2.shape == lambda2_width.shape
        if(np.all(lambda2_width != None)):
            A = create_faraday_matrix(lambda2, phi, lambda2_width, weights)
        else:
            A = create_faraday_matrix(lambda2, phi, lambda2 * 0., weights)
        A_H = np.conj(A.T)
        self.dir_op = lambda x: A @ x
        self.adj_op = lambda x: A_H @ x


class delay_operator:


    """
    Transform from delay to frequency
    """


    def __init__(self, nu, tau, weights):
        A = create_delay_matrix(nu, tau, weights)
        A_H = np.conj(A.T)
        self.dir_op = lambda x: A @ x
        self.adj_op = lambda x: A_H @ x


class stokes_operator:

    def __init__(self, nu, tau, lambda2, phi, weightsI, weightsP):
        self.faraday_op = faraday_operator(lambda2, phi, weightsP)
        self.delay_op = delay_operator(nu, tau, weightsI)
        self.i_size = len(nu)
        self.p_size = len(lambda2)
        self.tau_size = len(tau)
        self.phi_size = len(phi)

    def dir_op(self, x):
        out[:self.tau_size] = self.delay_op.dir_op(x[:self.i_size])
        out[self.tau_size:] = self.faraday_op.dir_op(x[self.i_size:])
        return out

    def adj_op(self, x):
        out[:self.i_size] = self.delay_op.dir_op(x[:self.tau_size])
        out[self.i_size:] = self.faraday_op.dir_op(x[self.tau_size:])
        return out


def phi_parameters(lambda2, lambda2_width):
    dphi = 1./np.max(lambda2)
    phi_max = np.max(np.abs(1./lambda2_width))
    phi_min = np.min(np.abs(1./lambda2_width))
    logger.info("\u03C6 resolution %s rad/m^2.", dphi)
    logger.info("\u03C6 max range is %s rad/m^2.", phi_max)
    logger.info("\u03C6 min range is %s rad/m^2.", phi_min)
    return phi_max, phi_min, dphi
