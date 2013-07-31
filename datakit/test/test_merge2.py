import sys
sys.path.append('..')
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

from merge import DbDataSource, FeatureCache, EventBuilder, make_cache

TEST_OUTPUT_FILE = 'test_merge_out_2.csv'


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
    views_cache = make_cache(
        'views_info', 'promotion_views'
        )

    print 'Initializing promotion cache...'
    promotion_cache.init_cache()
    print 'Initializing opener cache...'
    opener_cache.init_cache()
    viewer_cache.set_cache_like(opener_cache)
    print 'Initializing views cache...'
    views_cache.init_cache()

    return EventBuilder([('promotion_id', promotion_cache),
                         ('opener_id', opener_cache),
                         ('viewer_id', viewer_cache),
                         (["promotion_id", "viewer_id"], views_cache)
                         ])

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

    df.to_csv(TEST_OUTPUT_FILE, index=False, encoding='utf-8')


    # sort by promotion_id, user_id_viewer
    df = df.sort_index(by=['promotion_id', 'user_id_viewer'])
    df = df.set_index(np.arange(len(df)))

    return df


def main():
    df_test = get_test_df()

if __name__ == '__main__':
    main()
