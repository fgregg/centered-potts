import unittest

import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal

from pseudolikelihood.centered_potts import rpotts, grid_adjacency_matrix, CenteredPotts
import pseudolikelihood.centered_potts as cp

class BinaryPottsTest(unittest.TestCase) :
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

        self.Y_multi = potts.lbin.transform(Z)
        self.Z = Z

        self.avg_neighbors = A.sum(axis=1).mean()

    def test_fit(self):
        potts = CenteredPotts(C=float('inf'))
        potts.fit((self.X, self.A), self.Z)
        assert_array_almost_equal(potts.coef_,
                                  np.array([[2.386,  2.147, -2.888]]),
                                  3)
        assert_array_almost_equal(potts.intercept_,
                                  np.array([-0.076]),
                                  3)


    def test_gradient(self):
        w = np.array([[0, 1, 1, -0.5 * self.avg_neighbors]])
        A = self.A
        features = self.X
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss_grad(w, features, A, Y, alpha, sample_weight)
        loss, grad, p = out

        assert_array_almost_equal(grad,
                                  np.array([ 3.738, -6.917, -5.893,  5.798]),
                                  3)

    def test_loss(self):
        
        w = np.array([[0, 1, 1, -0.5 * self.avg_neighbors]])
        A = self.A
        features = self.X
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss(w, features, A, Y, alpha, sample_weight)
        loss, p_nonspatial, p, out_w = out

        assert_approx_equal(loss, 232.612, 3)
        assert_array_almost_equal(p_nonspatial.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(p.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(w, out_w)

    # def test_sampler(self):
    #     features = self.X
    #     A = self.A
    #     Z = self.Z

    #     potts = CenteredPotts(C=float('inf'))
    #     potts.fit((features, A), Z)

    #     target = [[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
    #                1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
    #                0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
    #                0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
    #                1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    #                1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1,
    #                0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
    #                0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
    #                0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
    #                1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
    #                1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
    #                1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #                0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
    #                0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
    #                0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,
    #                1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    #                0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    #                0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    #                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1,
    #                0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
    #                1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
    #                0, 1, 0, 1, 0, 0, 0, 0, 1]]

    #     import random
    #     random.seed(12345)
    #     np.random.seed(12345)
    #     assert_array_almost_equal(rpotts((features, A), potts).T,
    #                               np.array(target))
        
class MultiClassPottsTest(unittest.TestCase) :
    def setUp(self):
        n_rows, n_cols = 20, 20
        n_cells = n_rows * n_cols
        A = grid_adjacency_matrix(n_rows, n_cols)

        X = np.vstack((np.tile(np.arange(n_cols)/(n_rows - 1) - 0.5, n_rows),
                       np.repeat(np.arange(n_cols)/(n_rows - 1) - 0.5, n_rows))).T

        self.A, self.X = A, X

        np.random.seed(123)
        self.Z = np.random.randint(0, 3, n_cells)
        
        potts = CenteredPotts()
        potts.fit((X, A), self.Z)

        self.Y_multi = potts.lbin.transform(self.Z)
        self.avg_neighbors = A.sum(axis=1).mean()


    def test_loss(self):

        w = np.array([[0, 1, 1, -0.5, -0.5],
                      [0, 1, 1, -0.5, -0.5]])
        features = self.X
        A = self.A
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss(w, features, A, Y, alpha, sample_weight)
        loss, p_nonspatial, p, out_w = out

        assert_approx_equal(loss, 450.849, 3)
        assert_array_almost_equal(p_nonspatial.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(p.sum(axis=1), np.ones(features.shape[0]))
        assert_array_almost_equal(w, out_w)

    
    def test_gradient(self):
        w = np.array([[0, 1, 1, -0.5 * self.avg_neighbors, -0.5 * self.avg_neighbors],
                      [0, 1, 1, -0.5 * self.avg_neighbors, -0.5 * self.avg_neighbors]])
        features = self.X
        A = self.A
        Y = self.Y_multi
        alpha = 0
        sample_weight = np.ones(features.shape[0])

        out = cp._multinomial_loss_grad(w, features, A, Y, alpha, sample_weight)
        loss, grad, p = out

        assert_array_almost_equal(grad,
                                  np.array([ 15.079, 3.986, 5.829, -11.58 , -1.336,
                                              1.399,  3.909, 3.78, -11.725, -3.019]),
                                  2)
