"""Setup script for huest.
Installation command::
    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='multiscale_dense',

    version='0.0.0',

    url='https://github.com/adler-j/multiscale_dense',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    packages=find_packages(exclude=['*test*']),
    package_dir={'multiscale_dense': 'multiscale_dense'},

    install_requires=['numpy']
)