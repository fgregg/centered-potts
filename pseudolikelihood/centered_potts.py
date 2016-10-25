import numpy as np

from scipy import optimize
from scipy.sparse import coo_matrix

import sklearn.preprocessing
from sklearn.utils.extmath import (logsumexp, safe_sparse_dot, squared_norm,
                                   softmax)


class CenteredPotts(object):
    def __init__(self, tol=1e-4, max_iter=100, verbose=0, C=1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C

    def fit(self, X, y):
        features, A = X

        if len(y.shape) > 1 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        else:
            self.classes_ = np.unique(y)

        n_samples, n_features = features.shape

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
" class: %r" % classes_[0])

        sample_weight = np.ones(features.shape[0])

        if len(self.classes_) < 3 and (len(y.shape) < 2 or y.shape[1] < 2):
            self.lbin = BinaryLabelBinarizer()
        else:
            self.lbin = LabelBinarizer()
        Y_multi = self.lbin.fit_transform(y)

        w0 = np.zeros((self.classes_.size - 1, n_features + n_classes),
                      order='F')

        w0 = w0.ravel()
        target = Y_multi

        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]

        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(features, A, target, 1. / self.C, sample_weight),
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
        features, A = X

        Y_multi = self.lbin.transform(y)

        betas = self.coef_[:, :-1]
        eta = self.coef_[:, -1]

        p = safe_sparse_dot(features, betas.T, dense_output=True)
        p += self.intercept_

        p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
        p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
        p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

        spatial = safe_sparse_dot(A, (Y_multi - p_nonspatial))[:, :-1]

        p += eta.T * np.array(spatial/A.sum(axis=1))

        p = np.hstack((p, np.zeros((features.shape[0], 1))))

        return softmax(p)

    def neighbors_from_adjacency(self, A, y):
        Y_multi = self.lbin.transform(y)

        neighbors = A * Y_multi

        potts_neighbors = np.zeros((neighbors.shape[1], neighbors.shape[0], 2))
        for cls in range(neighbors.shape[1]):
            potts_neighbors[cls][:, 0] = neighbors[:, cls]
            potts_neighbors[cls][:, 1] = neighbors.sum(axis=1) - neighbors[:, cls]

        return potts_neighbors
    



def _multinomial_loss(w, features, A, Y, alpha, sample_weight):
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
    n_neighbors = A.sum(axis=1).reshape(-1, 1)

    n_observations = Y.sum(axis=1)[:, np.newaxis]

    w = w.reshape(n_classes - 1, -1)
    sample_weight = sample_weight[:, np.newaxis]
    intercept = w[:, 0]
    betas = w[:, 1:(n_features + 1)]
    eta = w[:, (n_features + 1):]

    p = safe_sparse_dot(features, betas.T)
    p += intercept

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

    spatial = safe_sparse_dot(A, (Y/n_observations - p_nonspatial))/n_neighbors
    spatial[np.isnan(spatial)] = 0

    for k in range(n_classes - 1):
        p[:, k:k+1] += safe_sparse_dot(np.delete(spatial, k, axis = 1),
                                       eta[k:k+1, :].T)
        
    
    p = np.hstack((p, np.zeros((features.shape[0], 1))))

    loss = (p * Y).sum(axis=1) - logsumexp(p, axis=1) * n_observations[:, 0]
    loss += log_multinomial_coefficient(Y)
    loss = -(loss * sample_weight[:, 0]).sum()
    loss += 0.5 * alpha * squared_norm(w[:, 1:])

    p -= logsumexp(p, axis=1)[:, np.newaxis]
    p = np.exp(p, p)

    return loss, p_nonspatial, p, w

def _multinomial_loss_grad(w, features, A, Y, alpha, sample_weight):
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
    n_neighbors = A.sum(axis=1).reshape(-1, 1)
    n_observations = Y.sum(axis=1)[:, np.newaxis]
    
    grad = np.zeros((n_classes - 1, n_features + n_classes))

    loss, p_nonspatial, p, w = _multinomial_loss(w, features, A, Y,
                                                 alpha, sample_weight)

    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p * n_observations - Y)[:, :n_classes-1]

    features = np.hstack((np.ones((features.shape[0], 1)), features))
    for k in range(n_classes - 1):
        mu = np.delete(p_nonspatial, k, axis=1)
        mu *= p_nonspatial[:, k][:, np.newaxis]

        centered_features = features.copy()
        for j, eta in enumerate(w[k, (n_features + 1):]):
            spatial = safe_sparse_dot(A, features * mu[:, j:j+1])/n_neighbors
            spatial[np.isnan(spatial)] = 0
            centered_features -= eta * spatial
        
        grad[k, :(n_features + 1)] = safe_sparse_dot(diff[:, k].T, centered_features)

        
    spatial = safe_sparse_dot(A, (Y/n_observations - p_nonspatial))/n_neighbors
    spatial[np.isnan(spatial)] = 0

    for k in range(n_classes - 1):
        grad[k, (n_features + 1):] = (diff[:, k:k+1]  *  np.array(np.delete(spatial, k, axis = 1))).sum(axis=0)


    grad[:, 1:] += alpha * w[:, 1:]

    return loss, grad.ravel(), p



def rpotts(X, model):
    features, A = X

    n_classes = len(model.classes_)
    n_sites = features.shape[0]
    n_neighbors = A.sum(axis=1).reshape(-1, 1)

    R = np.random.uniform(size=(n_sites, 1))

    lower = np.empty((n_sites, 1))
    upper = np.empty((n_sites, 1))

    betas = model.coef_[:, :-1]
    eta = model.coef_[:, -1:]

    p = safe_sparse_dot(features, betas.T, dense_output=True)
    p += model.intercept_

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

    _target = model.lbin.transform

    i = 0
    while not np.array_equal(upper, lower):
        R = np.hstack((np.random.uniform(size=R.shape), R))

        print(upper.sum(), lower.sum())
        lower[:] = 0
        upper[:] = n_classes - 1

        for r in R.T:
            r = r.reshape(-1, 1)

            upper_multi = _target(upper)
            upper_spatial = safe_sparse_dot(A, (upper_multi - p_nonspatial))[:, :-1]/n_neighbors
            upper_spatial[np.isnan(upper_spatial)] = 0

            upper_p = p + (eta.T * np.array(upper_spatial))
            upper_p = softmax(np.hstack((upper_p, np.zeros((features.shape[0], 1)))))
            upper_p = upper_p.cumsum(axis=1)
            
            lower_multi = _target(lower)
            lower_spatial = safe_sparse_dot(A, (lower_multi - p_nonspatial))[:, :-1]/n_neighbors
            lower_spatial[np.isnan(lower_spatial)] = 0

            lower_p = p + (eta.T * np.array(lower_spatial))
            lower_p = softmax(np.hstack((lower_p, np.zeros((features.shape[0], 1)))))
            lower_p = lower_p.cumsum(axis=1)

            upper = (upper_p > r).argmax(axis = 1)
            lower = (lower_p > r).argmax(axis = 1)

    return lower.reshape(-1, 1)

class BinaryLabelBinarizer(sklearn.preprocessing.LabelBinarizer):
    def transform(self, X):
        if np.in1d(X, [0, 1]).all():
            X = X.reshape(-1, 1)
            X = np.hstack((X, 1 - X))
        else:
            X = super().transform(X)

        return X

class LabelBinarizer(sklearn.preprocessing.LabelBinarizer):
    def fit(self, X, **fit_params):
        if len(X.shape) > 1 and X.shape[1] > 1:
            self.multi_output = True
        else:
            self.multi_output = False
            super().fit(X, **fit_params)

        return self

    def transform(self, X):
        if self.multi_output:
            return X
        else:
            return super().transform(X)
    
    

def to_adjacency(edges, N):
    A = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(N, N)).tocsc()
    A += A.T

    return A
        

def grid_adjacency_matrix(M, N=None):
    if N is None:
        N = M

    n_cells = N * M
    A = np.zeros((n_cells, n_cells))

    for i in range(1, n_cells + 1):
        up = i - N
        down = i + N
        left = i - 1
        right = i + 1
        if up > 0:
            A[i - 1, up - 1] = 1
        if down <= n_cells:
            A[i - 1, down - 1] = 1
        if left % N != 0:
            A[i - 1, left - 1] = 1
        if (right <= n_cells) and (i % N != 0):
            A[i - 1, right - 1] = 1

    return A

def log_factorial(i):
    return np.sum(np.log(np.arange(1, i + 1)))

def log_multinomial_coefficient(choices):
    m = np.sum(choices, axis=1)
    lf = np.vectorize(log_factorial)

    coef = lf(m) - np.sum(lf(choices))

    return coef
    

    
    
