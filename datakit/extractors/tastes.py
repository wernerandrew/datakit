# tastes.py
#
# A general catch-all for the 'categories' and 'interests' components 
# of the data in the system.

from itertools import izip

import numpy as np
import pandas as pd

from settings import get_category_map, get_interest_map
from util import safe_get_values

def expand_categories(categories):
    """The category mapping is a dict mapping numeric keys ("parents") 
    to lists of numeric keys ("children").

    Given a list of numbers, expand_categories returns: (1) a list 
    of numbers corresponding to "parent nodes", plus (2) a list of 
    (parent, child) tuples for any children in the list."""
    if expand_categories.parents is None:
        # initialize cached storage for quick lookup
        category_map = get_category_map()
        expand_categories.parents = set(map(int, category_map.keys()))
        expand_categories.children = {}
        for parent in category_map:
            for child in category_map[parent]:
                expand_categories.children[child] = (int(parent), child)
                
    category_parents = []
    category_children = []
    for cat in categories:
        if cat in expand_categories.parents:
            category_parents.append(cat)
        elif cat in expand_categories.children:
            category_children.append(expand_categories.children[cat])

    return (category_parents, category_children)

expand_categories.parents = None
expand_categories.children = None    

# strip whitespace and turn comma delimited list into list of integers
def make_int_list(s, delim=','):
    if not s or pd.isnull(s):
        return []
    else:
        s = s.replace(' ', '')
        return map(int, s.split(delim))


# calculate jaccard similarity of two sets s1, s2
# returns 0 if both empty
def jaccard_similarity(s1, s2):
    common = len(s1.intersection(s2))
    total = len(s1.union(s2))
    if total > 0:
        return float(common) / total
    else:
        return 0

class CategoryParentCommon(object):
    """Derive number of parent categories in common between two
    category lists."""
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        left, right = X.columns
        result = np.empty(len(X), dtype='float64')

        for i, (cat_a, cat_b) in enumerate(izip(X[left], X[right])):
            parents_a, _ = expand_categories(make_int_list(cat_a, ','))
            parents_b, _ = expand_categories(make_int_list(cat_b, ','))
            result[i] = len(set(parents_a).intersection(set(parents_b)))
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)

class CategoryParentJaccard(object):
    """Determine Jaccard similarity of two lists of parent categories."""
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        left, right = X.columns
        result = np.empty(len(X), dtype='float64')

        for i, (cat_a, cat_b) in enumerate(izip(X[left], X[right])):
            parents_a, _ = expand_categories(make_int_list(cat_a, ','))
            parents_b, _ = expand_categories(make_int_list(cat_b, ','))
            result[i] = jaccard_similarity(set(parents_a), set(parents_b))
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)

class CategoryChildCommon(object):
    """Derive number of child categories in common between two
    category lists."""
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        left, right = X.columns
        result = np.empty(len(X), dtype='float64')

        for i, (cat_a, cat_b) in enumerate(izip(X[left], X[right])):
            _, child_a = expand_categories(make_int_list(cat_a, ','))
            _, child_b = expand_categories(make_int_list(cat_b, ','))
            result[i] = len(set(child_a).intersection(set(child_b)))
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)

class CategoryChildJaccard(object):
    """Determine Jaccard similarity of two lists of parent categories."""
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        left, right = X.columns
        result = np.empty(len(X), dtype='float64')

        for i, (cat_a, cat_b) in enumerate(izip(X[left], X[right])):
            _, child_a = expand_categories(make_int_list(cat_a, ','))
            _, child_b = expand_categories(make_int_list(cat_b, ','))
            result[i] = jaccard_similarity(set(child_a), set(child_b))
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)
