import numpy as np


def create_fourier_matrix(lambda2, phi, lambda2_width):
    A = np.zeros((len(lambda2), len(phi)))
    for r in range(len(lambda2)):
        A[r, :] = np.exp(2 * lambda2[r] * phi * 1j) * np.sinc(2. * phi * lambda2_width[r]/np.pi)
    return A

class Faraday_operator:
    """
Operator that simulates a Faraday rotation observation and finite channel width.

INPUT:
    lambda2 - channel locations in m^2
    phi - pixel locations in rad/m^2
    lambda2_width - channel widths in m^2

"""


def __init__(lambda2, phi, lambda2_width = None):
    if(np.all(lambda2_width != None)):
        assert len(lambda2) == len(lambda2_width)
    if(np.all(lambda2_width != None)):
        A = create_fourier_matrix(lambda2, phi, lambda2_width)
    else:
        A = create_fourier_matrix(lambda2, phi, lambda2 * 0.)
    A_H = np.conj(A.T)
    self.dir_op = lambda x: A @ x
    self.adj_op = lambda x: A_H @ x
