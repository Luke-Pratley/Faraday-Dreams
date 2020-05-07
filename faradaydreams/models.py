import numpy as np
import scipy.integrate as integrate


def gaussian(phi, P, std, phi0, chi0):
    return P * np.exp(-(phi0 - phi)**2/(2 * std**2) + 1j * 2 * chi0)


def box(phi, P, width, phi0, chi0):
    model = np.zeros(phi.shape) * 1j
    logic_expr = np.abs(phi0 - phi) <= width/2.
    model[logic_expr] = np.exp(1j * 2 * chi0)
    return P * model


def delta(phi, P, phi0, chi0):
    return box(phi, P, np.abs(phi[0] - phi[1]), phi0, chi0)


def deltas(phi, Ps, phi0s, chi0s):
    model = np.zeros(phi.shape) * 1j
    for i in range(len(phi0s)):
        model += delta(phi, Ps[i], phi0s[i], chi0s[i])
    return model


def from_analytic_lambda2(QUlambda2_function, phi, a, b):
    result = phi * 0j
    for i in range(len(phi)):
        result[i] = integrate.quad(lambda lambda2: QUlambda2_function(lambda2) * np.exp(-2 * lambda2 * phi[i])/(2 * np.pi), a, b)
    return result
