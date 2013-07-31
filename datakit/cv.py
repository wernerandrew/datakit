# cv.py
# Cross validation and grid search routines meant to be applied
# to pipelines that take pandas objects.

import sys
import time # for random seeds

import numpy as np
import pandas as pd
from sklearn.grid_search import IterGrid
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def cv_dataframe(model, X, y, error_fn=mean_squared_error, 
                 score_fn=None, n_folds=5, verbose=True, 
                 predict_method='predict'):
    """
    Run k-fold cross validation over a training data set.
    Returns the mean error or score across all folds.
    
    Parameters:
    ----------
    model: A predictor implementing fit and predict information.

    X: training data that is capable of being indexed with a boolean
       array (e.g., DataFrame or 2D ndarray).

    y: target values to be fit by the model

    error_fn: defaults to mean_squared_error; takes precedence if no
       score_fn defined.  Should follow scikit-learn convention for
       argument order (y_true, y_pred)

    score_fn: if defined, takes precedence over error_fn.  Should follow
       scikit-learn convention for argument order (y_true, y_pred)

    n_folds: number of folds to use for K-fold cross validation

    verbose: if True, print status information during execution.

    predict_method: defaults to 'predict', which is fine except for 
       certain cases where an alternative prediction type is desired
       (e.g., predict_proba on certain classifiers)
    """
    if score_fn is not None:
        use_score = True
        result_type = 'score'
    else:
        use_score = False
        result_type = 'error'

    folds = KFold(len(X), n_folds=n_folds, indices=False, shuffle=True,
                  random_state=int(time.time()))
    results = []
    if verbose:
        print 'Performing cross validation...'
    for i, (train, test) in enumerate(folds):
        if verbose:
            sys.stdout.write('Fold {0}/{1}...'.format(i + 1, n_folds))
            sys.stdout.flush()
        
        model.fit(X[train], y[train])
        y_cv = y[test]
        yhat_cv = model.predict(X[test])

        if use_score:
            results.append(score_fn(y_cv, yhat_cv))
        else:
            results.append(error_fn(y_cv, yhat_cv))
            
        if verbose:
            print 'done! ({0}: {1})'.format(result_type, results[-1])

    final_result = np.mean(results)
    if verbose:
        print 'Final mean {0}: {1}'.format(result_type, final_result)

    return final_result

class DataFrameCV(object):
    """
    Stripped-down version of of GridSearchCV with the ability to work with
    DataFrames.  Does not yet support parallelism or a few of the more
    complex GridSearchCV features.
    """
    def __init__(self, model, params, n_folds=5, error_fn=mean_squared_error, 
                 score_fn=None, verbose=False):
        self.model = model
        self.params = params
        self.n_folds = n_folds
        self.error_fn = error_fn
        self.score_fn = score_fn
        self.verbose = verbose

        if self.score_fn is not None:
            self.use_score = True
        else:
            self.use_score = False

        self.best_result = None
        self.best_params = None

    def fit(self, X, y):
        self._reset_estimator_stats()
        param_iter = IterGrid(self.params)
        if self.verbose:
            print '*** Beginning cross validation grid search ***'
        for params in param_iter:
            if self.verbose:
                print 'Executing CV run:', params
            self.model.set_params(**params)
            result = cv_dataframe(self.model, X, y, self.error_fn, 
                                  self.score_fn, n_folds=self.n_folds,
                                  verbose=self.verbose)
            if self.use_score:
                # this means higher results are better
                if self.best_result is None or result > self.best_result:
                    self.best_result = result
                    self.best_params = params
            else:
                if self.best_result is None or result < self.best_result:
                    self.best_result = result
                    self.best_params = params

            if self.verbose:
                print 'Finished run; result = {0} (best = {1})'.format(
                    result, self.best_result
                    )
                
        if self.verbose:
            print '*** Complete ***'
            print 'Training best model...'
        self.model.set_params(**self.best_params)
        self.model.fit(X, y)
        if self.verbose:
            print 'Done! Best result: {0}, best params: {1}'.format(
                self.best_result, self.best_params
                )

    def predict(self, X):
        return self.model.fit(X, y)

    def _reset_estimator_stats(self):
        self.best_result = None
        self.best_params = None
        self.best_estimator = None
