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

# collection of utility functions

import datetime
import time
import calendar
import collections

import numpy as np
import pandas as pd

# extract the values from a DataFrame, if necessary, for use
# with certain feature mappers.
# otherwise, return unchanged.
def safe_get_values(X):
    if hasattr(X, 'values'):
        return X.values
    else:
        return X

# check if any values in an array are null:
def check_any_null(X):
    if isinstance(X, collections.Iterable):
        return any(pd.isnull(x) for x in X)
    else:
        return (X is None or pd.isnull(X))

# convert a datetime to epoch time
def convert_datetime_epoch(dt):
    return time.mktime(dt.timetuple())


# convert a date to epoch time
def convert_date_epoch(d):
    epoch = datetime.date(1970,1,1)
    return (d-epoch).days * 86400


# convert a datetime tuple to a date 
def convert_datetime_tuple(dtup):
    day = calendar.monthrange(dtup[0],dtup[1])[1]
    return datetime.date(dtup[0],dtup[1], day)

# decide is number or not
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# shorthand to print only if a "verbose" flag is set
def maybe_print(s, verbose=True):
    if verbose:
        print s
