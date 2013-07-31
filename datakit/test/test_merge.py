import sys
sys.path.append('..')
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

from merge import DbDataSource, FeatureCache, EventBuilder, make_cache

TEST_OUTPUT_FILE = 'test_merge_out.csv'
TEST_SQL_SCRIPT = 'test_merge_sql.sql'

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--use-stored-data', action='store_true')
    args = parser.parse_args()
    return args

def get_builder():
    promotion_cache = make_cache(
        'promotion_info', 'promotion_info', 
        convert_names={'id': 'promotion_id',
                      'cached_interest_list': 'promo_interest_list',
                      'cached_category_list': 'promo_category_list'}
        )
    opener_cache = make_cache(
        'opener_info', 'user_info', unique_code='opener'
        )
    viewer_cache = make_cache(
        'viewer_info', 'user_info', unique_code='viewer'
        )
    print 'Initializing promotion cache...'
    promotion_cache.init_cache()
    print 'Initializing opener cache...'
    opener_cache.init_cache()
    viewer_cache.set_cache_like(opener_cache)
    return EventBuilder([('promotion_id', promotion_cache), 
                         ('opener_id', opener_cache), 
                         ('viewer_id', viewer_cache)])

def get_test_df():
    print 'Getting pipeline...'
    builder = get_builder()
    sql = 'SELECT promotion_id, user_id AS viewer_id, status ' \
        'FROM promotion_approvals ' \
        'WHERE promotion_id > 200600 AND promotion_id < 200800'
    source = DbDataSource(sql)
    print 'Building events...'
    df = builder.build_events(source)
    print 'Done!'
    # sort by promotion_id, user_id_viewer
    df = df.sort_index(by=['promotion_id', 'user_id_viewer'])
    df = df.set_index(np.arange(len(df)))
    return df

def get_sql_df(use_stored=False):
    if use_stored:
        print 'Reading test data...'
        df = pd.read_csv(TEST_OUTPUT_FILE)
    else:
        print 'Rebuilding test data (this may take a while)...'
        sql = open(TEST_SQL_SCRIPT).read()
        src = DbDataSource(sql)
        result = list(src.iterrows())
        consolidated = defaultdict(list)
        for row in result:
            for k, v in row.iteritems():
                consolidated[k].append(v)
        df = pd.DataFrame(consolidated)
        print 'Saving test data...'
        df.to_csv(TEST_OUTPUT_FILE, index=False, encoding='utf-8')
    df = df.sort_index(by=['promotion_id', 'user_id_viewer'])
    df = df.set_index(np.arange(len(df)))
    return df

def assert_df_equal(x, y):
    assert(set(x.columns) == set(y.columns))
    try:
        # check numeric columns / zipcodes only for now
        # otherwise have some pandas weirdness for null values
        # possibly address this as a TODO
        check_cols = ['promotion_id',
                      'user_id_opener',
                      'user_id_viewer',
                      'total_fans_opener',
                      'total_fans_viewer']
        for col in check_cols:
            print 'Checking', col, '...'
            assert(all(x[col] == y[col]))
    except AssertionError as err:
        print 'Assertion error: {0}'.format(col)
        print x[col]
        print y[col]
        sys.exit(0)
    print 'Passed tests!'

def main(args):
    df_sql = get_sql_df(args.use_stored_data)
    df_test = get_test_df()
    assert_df_equal(df_sql, df_test)

if __name__ == '__main__':
    main(get_args())
