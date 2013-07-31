# Copyright (c) 2013 Andrew Werner and Anthony DeGangi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# some basic and useful extractors

from collections import defaultdict
from itertools import izip

import numpy as np
from sklearn.pipeline import Pipeline

from util import safe_get_values, check_any_null

class Binner(BaseEstimator):
    """Takes a single column vector of values and converts to bins.
    Uses the numpy.digitize function to do the heavy lifting.
    """
    def __init__(self, nbins=10, has_log_bins=False):
        self.nbins = nbins
        self.has_log_bins = has_log_bins
        self.bin_edges = None
        self._offset = 0

    def fit(self, X, y=None):
        self._offset = 0
        minval, maxval = None, None
        for x in X:
            if check_any_null(x):
                continue
            if minval is None:
                minval = x
                maxval = x
            elif x < minval:
                minval = x
            elif x > maxval:
                maxval = x
                
        if self.has_log_bins:
            if minval <= 0:
                self._offset = 1 - minval
                minval += self._offset
                maxval += self._offset
            self.bin_edges = np.logspace(np.log10(minval), 
                                         np.log10(maxval), 
                                         self.nbins)
        else:
            self.bin_edges = np.linspace(minval, maxval, self.nbins)

    def transform(self, X):
        X = safe_get_values(X)
        if self._offset > 0:
            X = X + self._offset
        return np.digitize(X, self.bin_edges)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class ValueCounter(BaseEstimator):
    """Given a matrix X, computes counting statistics using the 
    rows of the matrix as keys.

    For missing or null keys, the global population average is imputed.
    """
    def __init__(self, target_value=1):
        self.target_value = target_value
        self.reset_counts()
        self._columns = None # useful for dict conversions

    def reset_counts(self):
        self.hits = defaultdict(float)
        self.total = defaultdict(float)
        self.all_hits = 0.0
        self.grand_total = 0.0

    def get_probability(self, key):
        """Input: key for which conditional probability of a hit is to be 
        looked up.
        
        Output: float.  If key found, return observed conditional probability 
        of hit given the key passed.  Otherwise, return population probability 
        of hit."""
        if key in self.total:
            return self.hits.get(key, 0) / float(self.total[key])
        else:
            return self.all_hits / float(self.grand_total)

    def fit(self, X, y):
        self.reset_counts()
        X = safe_get_values(X)

        # keeping this variable local because instance method objects
        # can't e pickled
        key_factory = self._get_key_factory(X)

        for row, value in izip(X, y):
            hit_count = 1 if value == self.target_value else 0

            self.grand_total += 1
            self.all_hits += hit_count
            key = key_factory(row)

            if not check_any_null(key):
                self.total[key] += 1
                self.hits[key] += hit_count

    def transform(self, X):
        X = safe_get_values(X)

        key_factory = self._get_key_factory(X)
        result = np.empty(X.shape[0])
        for i, row in enumerate(X):
            key = key_factory(row)
            result[i] = self.get_probability(key)

        return result

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _get_key_factory(self, X):
        if X.ndim == 1:
            return self._identity
        else:
            return tuple

    def _identity(self, x):
        return x

def make_binned(extractor, nbins, has_log_bins=False):
    return Pipeline([('extractor', extractor),
                     ('binner', Binner(nbins, has_log_bins))])

def make_probability(extractor, target_value):
    return Pipeline([('extractor', extractor),
                     ('counter', ValueCounter(target_value))])
