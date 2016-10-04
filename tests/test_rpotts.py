import unittest

import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal

from pseudolikelihood.centered_potts import rpotts, grid_adjacency_matrix, CenteredPotts
import pseudolikelihood.centered_potts as cp

class RandomPottsTest(unittest.TestCase) :
    def setUp(self):
        n_rows, n_cols = 20, 20
        n_cells = n_rows * n_cols
        A = grid_adjacency_matrix(n_rows, n_cols)

        X = np.vstack((np.tile(np.arange(n_cols)/(n_rows - 1) - 0.5, n_rows),
                       np.repeat(np.arange(n_cols)/(n_rows - 1) - 0.5, n_rows))).T

        self.A, self.X = A, X

        Z = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                      1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                      1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                      0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
                      1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
                      0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,
                      1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                      0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1,
                      1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                      1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                      1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                      1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                      1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
                      0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                      1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1,
                      0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                      1, 1, 1, 0, 1, 0, 1, 1])

        potts = CenteredPotts()
        potts.fit((X, A), Z)

        rpotts((X, A), potts)
        self.Y_multi = potts.lbin.transform(Z)
        self.Z = Z

    def test_fit(self):
        potts = CenteredPotts()
        potts.fit((self.X, self.A), self.Z)
        print(potts.coef_)
        print(potts.intercept_)

    def test_gradient(self):
        w = np.array([[0, 1, 1, 0.5]])
        features = self.X
        A = self.A
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss_grad(w, features, A, Y, alpha, sample_weight)
        loss, grad, p = out

        assert_array_almost_equal(grad,
                                  np.array([  1.955, -3.102,  -1.857, -24.036,]),
                                  3)

    def test_loss(self):
        w = np.array([[0, 1, 1, 0.5]])
        features = self.X
        A = self.A
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss(w, features, A, Y, alpha, sample_weight)
        loss, p_nonspatial, p, out_w = out

        assert_approx_equal(loss, 231.7263)
        assert_array_almost_equal(p_nonspatial.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(p.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(w, out_w)
        
