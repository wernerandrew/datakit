#!/usr/bin/env python

from distutils.core import setup

setup(
    name='Datakit',
    version='0.1',
    packages=['datakit'],
    install_requires=[
        'numpy >= 1.6.0',
        'pandas >= 0.10.0',
        'scikit-learn >= 0.13.0'
    ]
)
