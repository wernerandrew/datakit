from collections import defaultdict
from itertools import izip

import numpy as np
import pandas as pd

from dbwrapper import DbConn

class DbDataSource(object):
    """
    Provides a iterator interface to a SQL query with automatic
    connection cleanup.
    """
    def __init__(self, sql, host, user, password, db, **kwargs):
        self.sql = sql
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.connect_args = kwargs

    def __iter__(self):
        return self.iterrows()

    def iterrows(self, args=None):
        with DbConn(self.host, self.user, self.password, self.db, 
                    connect_args=self.connect_args) as connection:
            cur = connection.execute(self.sql, args)
            while True:
                result = cur.fetchone()
                if result:
                    yield result
                else:
                    break

class CSVDataSource(object):
    """
    Wraps the pandas TextParser interface
    """
    def __init__(self, filename, chunksize=10000, **kwargs):
        """
        Parameters:
        ----------
        filename: name of file to be iterated through.

        kwargs (if any) are passed on to pandas.read_csv.
        """
        self.filename = filename
        self.read_csv_kwargs = kwargs
        self.read_csv_kwargs['chunksize'] = chunksize

    def __iter__(self):
        return self.iterrows()

    def iterrows(self):
        reader = pd.read_csv(self.filename, **self.read_csv_kwargs)
        try:
            while True:
                # yield first row as dict
                data = reader.get_chunk()
                for _, row in data.iterrows():
                    yield dict(row)
        except StopIteration:
            pass

class FeatureCache(object):
    def __init__(self, name, source, key, convert_names=None,
                 unique_code=None):
        """
        Parameters:

        name: A name for the cache object
        source: A data source, which can be any object that exposes an
                iterrows() method that returns an iterator to a series of
                dict-like object.  (Either a DbDataSource or a PyTables Table
                satisfies this requirement)
        key: A column name (expressed as a string) or a list of column
             names in the data source to use as a key.
        convert_names: Optional, a dict giving replacement names for the
                output.
        unique_code: Optional, used to disambiguate between different versions
                     of the same information (e.g., user IDs)

        Example:
        sql = 'SELECT id, genre FROM users'
        cache = FeatureCache('GenreMapping', DbDataSource(sql),
                             'id', convert_names={'id': 'user_id'})
        """
        self.name = name
        self.source = source
        self.key = key

        self.convert_names = convert_names
        self.unique_code = unique_code
        self.db_cols = None
        self.output_names = None

        if isinstance(self.key, basestring):
            self.single_key = True
        else:
            self.single_key = False

        self.cache = {}

    def init_cache(self):
        """
        Initialize the cache state based on the provided data source.
        """
        self.cache = {}
        row = None
        if self.single_key:
            for row in self.source.iterrows():
                key = row[self.key]
                value = tuple(row.values())
                self.cache[key] = value
        else:
            for row in self.source.iterrows():
                key = tuple(row[col] for col in self.key)
                value = tuple(row.values())
                self.cache[key] = value

        if row is not None:
            self._update_name_mapping(row)

    def set_cache_like(self, other_cache):
        """Useful when cached values overlap."""
        self.cache = other_cache.cache
        self._raw_output_names = list(other_cache._raw_output_names)
        self.output_names = self._maybe_add_suffix(self._raw_output_names)

    def transform_one(self, feature_key, x, copy=False):
        """
        Modify input by adding cached elements and 
        
        Parameters
        ----------
        feature_key : string or tuple
           Key or keys in the provided dict that should be used for the
           cache lookup.
        x : dict-like
           Data to be transformed.
        copy : boolean
           If True, input data is copied before modification.  If False,
           input data is modified in place.
        """
        if copy:
            x = x.copy()
        if self.single_key:
            key = x[feature_key]
            keys = [feature_key]
        else:
            key = tuple(x[col] for col in feature_key)
            keys = feature_key

        rowdata = self.cache.get(key, None)
        if rowdata:
            for col, data in izip(self.output_names, rowdata):
                if col not in keys:  #dont include the column more than once
                    x[col] = data
        else:
            for col in self.output_names:
                if col not in keys:
                    x[col] = np.nan

        return x


    def _update_name_mapping(self, row):
        # Set name mapping given a row from the database
        self._raw_output_names = row.keys()

        if self.convert_names is not None:
            for i, old_key in enumerate(self._raw_output_names):
                if old_key in self.convert_names:
                    self._raw_output_names[i] = self.convert_names[old_key]

        self.output_names = self._maybe_add_suffix(self._raw_output_names)

    def _maybe_add_suffix(self, names):
        # If a unique code is defined, add to list of names.
        # If not, return names unchanged.
        if self.unique_code is not None:
            suffix = '_{0}'.format(self.unique_code)
            return [k + suffix for k in names]
        else:
            return list(names)

class EventBuilder(object):
    """
    Given a data source, applies a series of FeatureCache transformations
    and returns a pandas DataFrame incorporating all modifications.
    """
    def __init__(self, mappers):
        """
        Parameters:
        ----------
        mappers: A list of (key, feature_cache) tuples.  The key is
           provided as a parameter to the transform method feature cache
           when assembling the event.
        """
        self.mappers = mappers

    def build_events(self, X):
        """
        Apply FeatureCache transformations to provided data.

        Parameters
        ---------
        X : iterable of dict-like or dict like
            Data to be transformed.
        """
        result = defaultdict(list)

        for xt in self.iter_events(X):
            for key, val in xt.iteritems():
                result[key].append(val)

        return pd.DataFrame(result)

    def iter_events(self, X):
        """
        Iterator interface to build_events.

        Parameters
        ----------
        X : iterable of dict-like or dict-like
            Data to be transformed.
        """
        if isinstance(X, dict):
            X = [X]
        for x in X:
            xt = x
            for cache_key, cache in self.mappers:
                xt = cache.transform_one(cache_key, xt)
            yield xt

def make_cache_from_db(name, db_table, db_columns, key,
                       where=None, group_by=None):
    """
    Convenience function to create a cache from a database.
    kwargs are passed to FeatureCache constructor.  In the event
    of a conflict with the settings file, the explicitly provided
    kwargs win the day.
    
    Parameters
    ----------
    name : string
       name of the new cache object.
    cache_type : string
       type of join to use, as defined in settings.py file
    """
    sql = _make_cache_query(db_table, db_columns, where=where,
                            group_by=group_by)
    source = DbDataSource(sql)
    cache = FeatureCache(name, source, key, **kwargs)
    return cache

def _make_cache_query(table, columns, where=None, group_by=None):
    # helper function to generate SQL for cache generation query
    sql = 'SELECT {0} FROM {1}'.format(','.join(columns), table)
    if where is not None:
        sql += ' WHERE ' + where
    if group_by is not None:
        sql += ' GROUP BY ' + ', '.join(group_by)
    return sql
