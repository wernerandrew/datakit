import sys
sys.path.append('..')
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mapper import FeatureMapper
from extractors import Binner, ValueCounter, make_binned
from zipcodes import get_distance_calculator
from tastes import CategoryParentCommon, CategoryParentJaccard, \
    CategoryChildCommon, CategoryChildJaccard
from settings import get_postcode_args

TEST_INPUT_FILE = 'test_merge_out.csv'
TEST_OUTPUT_FILE = 'test_map_out.csv'

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def get_test_data():
    # several columns need to be explicitly made into strings
    # this is only an issue when reading from a dataframe; in the live
    # setting, the data will come off the database as a string.
    # For some reason, pd.read_csv seems to want to turn "1,2,3" into a 
    # list...go figure.
    converters = {
        'category_list_opener': str,
        'category_list_viewer': str,
        'interest_list_opener': str,
        'interest_list_viewer': str,
        'promo_category_list': str,
        'promo_interest_list': str
        }
    return pd.read_csv('test_merge_out.csv', converters=converters)

def get_mapper():
    mapper = FeatureMapper(
        [('Distance', ['zipcode_opener', 'zipcode_viewer'], 
          get_distance_calculator()),
         ('LogBinDistance', ['zipcode_opener', 'zipcode_viewer'],
          make_binned(get_distance_calculator(), 20, has_log_bins=True)),
         ('PairProbability', ['user_id_opener', 'user_id_viewer'],
          ValueCounter(1.0)),
         ('ParentCommon', ['category_list_opener', 'category_list_viewer'],
          CategoryParentCommon()),
         ('ParentJaccard', ['category_list_opener', 'category_list_viewer'],
          CategoryParentJaccard()),
         ('ChildCommon', ['category_list_opener', 'category_list_viewer'],
          CategoryChildCommon()),
         ('ChildJaccard', ['category_list_opener', 'category_list_viewer'],
          CategoryChildJaccard())]
        )
    return mapper

def get_objective(df):
    y = ((df['status'] == 'accepted') | 
         (df['status'] == 'posted')).astype('float64')
    return y

def run_tests():
    df = get_test_data()
    mapper = get_mapper()
    y = get_objective(df)
    Xt = mapper.fit_transform(df, y)
    ncol = Xt.shape[1]
    df_out = pd.DataFrame(Xt, columns=list(alphabet[:ncol]))
    df_out.to_csv(TEST_OUTPUT_FILE, index=False, encoding='utf-8')

if __name__ == '__main__':
    run_tests()
