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
