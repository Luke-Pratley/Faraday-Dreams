import pytest
from numpy import linalg as LA
import numpy as np
import faradaydreams.measurement_operator

np.random.seed(42)

def test_faraday_operator():
    lambda2 = np.linspace(0, 6, 20)
    weights = np.ones((20))
    dphi = 0.24
    Nd = 16
    phi = faradaydreams.measurement_operator.create_phi_array(Nd, dphi)

    matrix_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=False)
    nufft_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=True)

    dir_inp = np.zeros((phi.shape[0]))
    dir_inp[8] = 1
    output1 = matrix_op.dir_op(dir_inp)
    output2 = nufft_op.dir_op(dir_inp)
    assert(np.allclose(output1, 1., 1e-2))
    assert(np.allclose(output2, 1., 1e-2))

    dir_inp = np.zeros((phi.shape[0]))
    dir_inp[9] = 1
    output1 = matrix_op.dir_op(dir_inp)
    output2 = nufft_op.dir_op(dir_inp)
    expected = np.exp(2j * lambda2 * phi[9])
    assert(np.allclose(output1, expected, 1e-3))
    assert(np.allclose(output2, expected, 1e-3))