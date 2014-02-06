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

from itertools import izip

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class FeatureMapper(BaseEstimator):
    """
    Class for aggregating feature generation on a DataFrame.
    Given a set of mappers, which are defined to operate on specified columns
    of the provided input pandas DataFrame, transforms data into a NumPy
    ndarray suitable to input to a scikit-learn object.
    """

    @classmethod
    def from_rules(cls, metadata, rules):
        """
        Uses a rules-based approach to extract features.
        - Column names are mapped to metadata, which is an arbitrary
          hashable python object (e.g., a tuple or namedtuple).

        - Rules are defined in terms of a mapping:
          metadata -> (MapperClass, [args], {kwargs})
          where the kwargs dict is optional.
        """
        features = []
        for col, md in metadata.iteritems():
            try:
                mapper_params = rules[md]
            except KeyError:
                msg = 'Col %s has unknown metadata' % col
                raise RuntimeError(msg)
            mapper = cls._make_mapper(params)
            mapper_name = '%s-%s' % (col, mapper.__name__)
            features.append((mapper_name, col, mapper))

        return cls(features)

    @staticmethod
    def _make_mapper(params):
        if len(params) == 2:
            mapper, args = params
            kwargs = {}
        else:
            mapper, args, kwargs = params

        return mapper(*args, **kwargs)

    def __init__(self, features):
        """
        Parameters
        ----------
        features : list of (name, columns, mapper) tuples
           Defines feature mapping for use.
           "name" defines the name of the mapper object.
           "columns" can be a string or a list, and defines the columns
           on which the mapper works.
           "mapper" is an instance of an object implementing the fit,
           transform, and fit_transform methods specified by the scikit-learn
           interface.
        """
        self.features = features
        self.named_features = {name: mapper 
                               for name, _, mapper in self.features}
        super(FeatureMapper, self).__init__()

    def fit(self, X, y=None):
        """
        Fit individual extractors to data.
        """
        for feature_name, columns, extractor in self.features:
            extractor.fit(X[columns], y)

    def transform(self, X):
        """
        Transform input data by extractors.
        """
        extracted = []
        for feature_name, columns, extractor in self.features:

            feature = extractor.transform(X[columns])

            if hasattr(feature, 'toarray'):
                feature = feature.toarray()
            if feature.ndim == 1:
                feature = feature.reshape((len(feature), 1))
            extracted.append(feature)

        if len(extracted) > 0:
            result = np.concatenate(extracted, axis=1)
        else:
            result = extracted[0]

        return result

    def fit_transform(self, X, y=None):
        """
        Fit mappers to data, then return transformed version of data.
        Generally used during the process of training a model.
        """
        extracted = []

        for feature_name, column_names, extractor in self.features:
            feature = extractor.fit_transform(X[column_names], y)
            if hasattr(feature, 'toarray'):
                feature = feature.toarray()
            if feature.ndim == 1:
                feature = feature.reshape((len(feature), 1))
            extracted.append(feature)

        if len(extracted) > 0:
            result = np.concatenate(extracted, axis=1)
        else:
            result = extracted[0]

        return result

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureMapper, self).get_params(deep=False)
        else:
            out = self.named_features.copy()
            for name, feature in self.named_features.iteritems():
                for key, value in feature.get_params(deep=True).iteritems():
                    out['{0}__{1}'.format(name, key)] = value
            return out
