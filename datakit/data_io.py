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

import os
import glob
import pickle
from datetime import date

def save_model(model, model_dir='model_cache'):
    filename = _get_next_model_path()
    with open(filename, 'w') as ofile:
        pickle.dump(model, ofile)

def load_newest_model(model_dir='model_cache'):
    model_files = _find_models()
    if model_files:
        version = max(model_files.keys())
        filename = model_files[version]
        return _load_model_file(filename)
    else:
        return None

def load_model_version(version, model_dir='model_cache'):
    model_files = _find_models(model_dir)
    if model_files:
        filename = model_files[version]
        return _load_model_file(filename)
    else:
        return None

def _get_next_model_path(model_dir='model_cache'):
    if os.path.exists(model_dir):
        if not os.path.isdir(model_dir):
            raise RuntimeError('non-directory specified for model path')
    else:
        os.makedirs(model_dir)

    model_files = _find_models(model_dir)
    if model_files:
        version = max(model_files.keys()) + 1
    else:
        version = 1
    today = date.today().isoformat()
    filename = 'model_{0}_{1}.bin'.format(version, today)
    fullpath = os.path.join(model_dir, filename)
    return fullpath

def _load_model_file(filename):
    with open(filename) as ifile:
        return pickle.load(ifile)

def _find_models(model_dir='model_cache'):
    model_pattern = os.path.join(get_model_path(), 'model_*_*.bin')
    model_paths = glob.glob(model_pattern)
    names = [os.path.splitext(os.path.basename(f))[0] 
             for f in model_paths]
    models = {}
    for name, path in zip(names, model_paths):
        _, version_str, _ = name.split('_')
        version = int(version_str)
        models[version] = path
    return models
