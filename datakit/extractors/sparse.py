# sparsify.py

import numpy as np
import scipy as sp
import scipy.sparse

from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator

class SparseIndicator(BaseEstimator):
    """
    Given an input vector X, where each entry is a delimited string, 
    returns a sparse indicator matrix for each element in the vector.
    
    Output matrix has rows equal to dimension of X
    """
    def __init__(self, delim=','):
        """
        Parameters:
        ----------
        delim: delimiter to use when splitting each element string
        """
        self.delim = delim
        self.id_lookup = {}
        self.name_lookup = {}
        self.ncol = 0
        
    def fit(self, X, y=None):
        self.id_lookup = {}
        self.name_lookup = {}
        self.ncol = 0
        current_id = 0
        
        for i, x in enumerate(X):
            tokens = x.split(self.delim)
            for tok in tokens:
                if tok not in self.id_lookup:
                    self.id_lookup[tok] = current_id
                    self.name_lookup[current_id] = tok
                    current_id += 1

        self.ncol = current_id
        return self

    def transform(self, X):
        rows = []
        cols = []
        vals = []

        for i, x in enumerate(X):
            tokens = x.split(self.delim)
            for tok in tokens:
                if tok in self.id_lookup:
                    rows.append(i)
                    cols.append(self.id_lookup[tok])
                    vals.append(1)

        nrow = X.shape[0]
        return sp.sparse.coo_matrix(
            (np.array(vals, dtype='float64'), (rows, cols)),
            shape=(nrow, self.ncol))

    def fit_transform(self, X, y=None):
        self.id_lookup = {}
        self.name_lookup = {}
        self.ncol = 0
        current_id = 0
        rows = []
        cols = []
        vals = []

        for i, x in enumerate(X):
            tokens = x.split(self.delim)
            for tok in tokens:
                if tok not in self.id_lookup:
                    self.id_lookup[tok] = current_id
                    self.name_lookup[current_id] = tok
                    current_id += 1
                rows.append(i)
                cols.append(self.id_lookup[tok])
                vals.append(1)

        nrow = X.shape[0]
        self.ncol = current_id
        return sp.sparse.coo_matrix(
            (np.array(vals, dtype='float64'), (rows, cols)),
            shape=(nrow, self.ncol))

    def to_name(self, X, default=None):
        try:
            return [self.name_lookup.get(x, default) for x in X]
        except TypeError:
            # catch situation where X is not iterable
            return self.name_lookup.get(x, default)

class SparseSelector(BaseEstimator):
    def __init__(self, alpha=1.0, fit_intercept=True, 
                 normalize=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.lasso = None

    def fit(self, X, y):
        self.lasso = Lasso(alpha=self.alpha, 
                           fit_intercept=self.fit_intercept,
                           normalize=self.normalize)
        self.lasso.fit(X, y)
        return self
        
    def transform(self, X):
        cols = np.nonzero(self.lasso.sparse_coef_)[1]
        if sp.sparse.issparse(X):
            return X.tocsc()[:, cols]
        else:
            return X[:, cols]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
