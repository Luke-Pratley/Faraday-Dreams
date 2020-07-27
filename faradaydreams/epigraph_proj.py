import numpy as np

class epi_proj:

    def __init__(self, indices1, indices2):
        self.beta = 1.
        self.indices1 = indices1
        self.indices2 = indices2

    def prox(self, x, tau):
        x1 = x[self.indices1]
        x2 = x[self.indices2]
        x[self.indices1][x1 < -x2] = 0.5 * (x1 - x2)
        x[self.indices2][x1 < -x2] = 0.5 * (x2 - x1)
        return x

    def fun(self, x):
        return 0.

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x


class epi_l1_proj:

    def __init__(self, indices1, indices2):
        self.beta = 1.
        self.indices1 = indices1
        self.indices2 = indices2

    def prox(self, x, tau):
        x1 = x[self.indices1]
        x2 = x[self.indices2]
        x[self.indices1][np.abs(x1) < -x2] = 0
        x[self.indices2][np.abs(x1) < -x2] = 0
        x[self.indices1][np.abs(x1) > x2] = x1[np.abs(
            x1) > x2] * 0.5 * (1 + x2[np.abs(x1) > x2]/np.abs(x1[np.abs(x1) > x2]))
        x[self.indices2][np.abs(x1) > x2] = np.abs(
            x1[np.abs(x1) > x2]) * 0.5 * (1 + x2[np.abs(x1) > x2]/np.abs(x1[np.abs(x1) > x2]))
        return x

    def fun(self, x):
        return 0.

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x
