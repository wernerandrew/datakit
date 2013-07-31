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

import sys
sys.path.append('..')

from sklearn.pipeline import Pipeline
import sklearn.metrics

import cv
from util import maybe_print

def get_pipeline(feature_set, predictor):
    return Pipeline([('mapper', feature_set), ('predictor', predictor)])

def choose_eval_fn(error_arg, score_arg):
    error, score = None, None
    if score_arg is None:
        if error_arg is None:
            error = sklearn.metrics.mean_squared_error
        else:
            error = getattr(sklearn.metrics, error_arg)
    else:
        score = getattr(sklearn.metrics, score_arg)

    return error, score

def parse_parameters(params):
    return {'predictor__{0}'.format(p): params[p] for p in params}

def setup_model(source, builder, objective_fn, verbose=True):
    maybe_print('Building events...', verbose)
    X = builder.build_events(source)
    maybe_print('Calculating objective...', verbose)
    y = objective_fn(X)
    return (X, y)

def train_model(model, X, y, parameters=None, verbose=True):
    if parameters is None:
        parameters = {}
    else:
        parameters = parse_parameters(parameters)

    model.set_params(**parameters)
    maybe_print('Fitting model...', verbose)
    model.fit(X, y)
    maybe_print('Done!', verbose)
    return model

def test_model(model, X, y, error_fn=None, score_fn=None, 
               search_params=None, verbose=True):

    maybe_print('Training model...', verbose)
    if search_params is not None:
        validator = cv.DataFrameCV(model, search_params, 
                                   n_folds=args.nfolds,
                                   error_fn=error_fn,
                                   score_fn=score_fn,
                                   verbose=verbose)
        validator.fit(X, y)
        err = validator.best_result
    else:
        err = cv.cv_dataframe(model, X, y, err_fn, score_fn, 
                              n_folds=args.nfolds, verbose=verbose)
    return err

def main(args):
    print 'Test complete! (result = {0})'.format(err)

if __name__ == '__main__':
    main(get_args())
