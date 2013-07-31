# collection of utility functions adapted from Clint's stuff.

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
