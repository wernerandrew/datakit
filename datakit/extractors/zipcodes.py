import re
from collections import defaultdict
from itertools import izip
from math import degrees, radians, sin, cos, acos

import numpy as np
import pandas as pd

from settings import get_postcode_args
from util import is_number, safe_get_values, check_any_null

def normalize_string(s):
    # strip whitespace and dashes, convert to uppercase
    s = s.replace(" ", "")
    s = s.replace("-", "")
    s = s.upper()
    return s

class PostMapperBase(object):
    """Handles file I/O and defines the basic interface for any
    postcode mapper objects."""
    def __init__(self, csvfile, code_col, lat_col, lng_col):
        self.csvfile = csvfile
        self.code_col = code_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.code_map = {}
    
    def init_cache(self):
        self.code_map = {}
        converters = {self.code_col: str} # store postcodes as strings
        df = pd.read_csv(self.csvfile, 
                         usecols=[self.code_col, self.lat_col, self.lng_col],
                         converters=converters)
        for _, row in df.iterrows():
            code = row[self.code_col]
            if code:
                norm_code = self.normalize(code)
                coords = (row[self.lat_col], row[self.lng_col])
                self.add_code(norm_code, coords)

    def add_code(self, norm_code, coords):
        self.code_map[norm_code] = coords

    def lookup(self, code):
        raise NotImplementedError()

    def normalize(self, code):
        raise NotImplementedError()

class AreaPostMapper(PostMapperBase):
    """
    Used where there is not a one-to-one correspondence between
    a normalized postcode and a latitude/longitude.  In this case,
    the derived value is the average of matching postcodes, of which
    there may be many.
    """
    def init_cache(self):
        # this is perhaps a bit kludgy
        self._counts = defaultdict(float)
        super(AreaPostMapper, self).init_cache()

    def add_code(self, norm_code, coords):
        lat, lng = coords
        if not (pd.isnull(lat) or pd.isnull(lng)):
            N = self._counts[norm_code]
            old_lat, old_lng = self.code_map.get(norm_code, (0, 0))
            new_lat = (N*old_lat + lat) / (N + 1.0)
            new_lng = (N*old_lng + lng) / (N + 1.0)
            self.code_map[norm_code] = (new_lat, new_lng)
            self._counts[norm_code] += 1.0
    
    def lookup(self, code):
        norm_code = self.normalize(code)
        return self.code_map.get(norm_code, None)

class USPostMapper(PostMapperBase):
    def lookup(self, code):
        norm_code = self.normalize(code)
        if is_number(norm_code):
            norm_code = norm_code[:5]
            return self.code_map.get(norm_code, None)
        else:
            return None

    def normalize(self, code):
        code = normalize_string(code)
        return code

class UKPostMapper(AreaPostMapper):
    def normalize(self, code):
        tokens = code.split()
        if tokens:
            return tokens[0].upper()
        else:
            return ''

class CanadaPostMapper(AreaPostMapper):
    def normalize(self, code):
        return code.upper()[:3]

class DistanceCalculator(object):
    """
    Given a dataframe X, having two columns, each being a postcode,
    calculate the distance between those post codes in miles.
    """
    def __init__(self, mappers, bin_function=None):
        self.mappers = mappers
        # initialize mappers once - save time curing cross validation
        # folds
        self._initialized = False
        
    def fit(self, X, y=None):
        if not self._initialized:
            for mapper in self.mappers:
                mapper.init_cache()
            self._initialized = True

    def transform(self, X):
        X = safe_get_values(X)

        result = np.empty(X.shape[0])
        for i, row in enumerate(X):
            if check_any_null(row):
                result[i] = np.nan
            else:
                code_a, code_b = row
                coord_a = self._get_coordinate(code_a)
                coord_b = self._get_coordinate(code_b)
                if check_any_null((coord_a, coord_b)):
                    result[i] = np.nan
                else:
                    result[i] = self._calc_dist(coord_a, coord_b)
        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _get_coordinate(self, code):
        for mapper in self.mappers:
            coord = mapper.lookup(code)
            if coord is not None:
                return coord
        return None

    def _calc_dist(self, a, b):
        # calculate the distance in statute miles based on coordinates
        lat_a, long_a = a
        lat_b, long_b = b
        
        lat_a = radians(lat_a)
        lat_b = radians(lat_b)
        long_diff = radians(long_a - long_b)
        distance = (sin(lat_a) * sin(lat_b) +
                    cos(lat_a) * cos(lat_b) * cos(long_diff))
        try:
            dist = degrees(acos(distance)) * 69.09 # magic number?
        except ValueError:
            dist = np.nan
            #print distance, lat_a, long_a, lat_b, long_b
        return dist

def get_distance_calculator():
    post_mappers = [USPostMapper(*get_postcode_args('US')),
                    UKPostMapper(*get_postcode_args('UK')),
                    CanadaPostMapper(*get_postcode_args('Canada'))]
    return DistanceCalculator(post_mappers)
