import numpy as np

from scipy import optimize

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import (logsumexp, safe_sparse_dot, squared_norm,
                                   softmax)


class CenteredPotts(object):
    def __init__(self, tol=1e-4, max_iter=100, verbose=0, C=1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C

    def fit(self, X, y):
        features, edges = X

        neighbors = neighbors_from_edges(edges, y)

        self.classes_ = np.unique(y)
        n_samples, n_features = features.shape

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
" class: %r" % classes_[0])

        sample_weight = np.ones(features.shape[0])

        lbin = LabelBinarizer()
        Y_multi = lbin.fit_transform(y)
            
        w0 = np.zeros((self.classes_.size - 1, n_features + 2),
                      order='F')

        w0 = w0.ravel()
        target = Y_multi

        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]

        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(features, neighbors, target, 1. / self.C, sample_weight),
            iprint=(self.verbose > 0) - 1, pgtol=self.tol, maxiter=self.max_iter)

        if info["warnflag"] == 1 and self.verbose > 0:
            warnings.warn("lbfgs failed to converge. Increase the number "
                          "of iterations.")

        n_iter_i = info['nit'] - 1

        self.coef_ = np.reshape(w0, (self.classes_.size - 1, -1))

        self.intercept_ = self.coef_[:, 0]
        self.coef_ = self.coef_[:, 1:]
        

        return self
        
        
    def predict_proba(self, X, y):
        features, edges = X
        neighbors = neighbors_from_edges(edges, y)

        lbin = LabelBinarizer()
        Y_multi = lbin.fit_transform(y)

        betas = self.coef_[:, :-1]
        eta = self.coef_[:, -1]

        scores = safe_sparse_dot(features, betas.T, dense_output=True)
        scores += self.intercept_
        scores += eta * ((1 - Y_multi) * neighbors).sum()

        scores = np.hstack((scores, np.zeros((features.shape[0], 1))))
        return softmax(scores)



def _multinomial_loss(w, features, neighbors, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = features.shape[1]
    w = w.reshape(n_classes - 1, -1)
    sample_weight = sample_weight[:, np.newaxis]
    intercept = w[:, 0]
    w = w[:, 1:]
    betas = w[:, :-1]
    eta = w[:, -1:]
    p = safe_sparse_dot(features, betas.T)
    p += intercept

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)[:, :-1]

    spatial = (neighbors[:-1, :, 0] * (0 - p_nonspatial).T +
               neighbors[:-1, :, 1] * (1 - p_nonspatial).T)
    p += (eta * spatial).T

    p = np.hstack((p, np.zeros((features.shape[0], 1))))
    p -= logsumexp(p, axis=1)[:, np.newaxis]

    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p_nonspatial, p, w


def _multinomial_loss_grad(w, features, neighbors, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    features : {array-like, sparse matrix}, shape (n_samples, n_features)
               Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = features.shape[1]
    grad = np.zeros((n_classes - 1, n_features + 2))
    loss, p_nonspatial, p, w = _multinomial_loss(w, features, neighbors, Y,
                                                 alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)[:, :n_classes-1]
    grad[:, 1:n_features + 1] = safe_sparse_dot(diff.T, features)

    spatial = (neighbors[:-1, :, 0] * (0 - p_nonspatial).T +
               neighbors[:-1, :, 1] * (1 - p_nonspatial).T)
    
    grad[:, -1] = (spatial * diff.T).sum(axis=1)
    grad[:, 1:n_features + 2] += alpha * w
    grad[:, 0] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def neighbors_from_edges(edges, y):
    # assumes that edges are unique and only exist once i.e. if 1, 0
    # appears then 0, 1 does not appear
    lbin = LabelBinarizer()
    Y_multi = lbin.fit_transform(y)

    neighbors = np.zeros_like(Y_multi)

    for i, j in edges:
        neighbors[i] += Y_multi[j]
        neighbors[j] += Y_multi[i]

    potts_neighbors = np.zeros((3, neighbors.shape[0], 2))
    for cls in range(neighbors.shape[1]):
        potts_neighbors[cls][:, 0] = neighbors[:, cls]
        potts_neighbors[cls][:, 1] = neighbors.sum(axis=1) - neighbors[:, cls]

    return potts_neighbors
