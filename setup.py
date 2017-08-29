# -*- coding: utf-8 -*-
"""
Setup file for the tlseparation package.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="tls_occlusion",
    version="1.0",
    author='Matheus Boni Vicari',
    author_email='matheus.boni.vicari@gmail.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['matmap2facets=tls_occlusion.command_line:matmap2facets',
                            'convert_hips=tls_occlusion.command_line:convert_hips',
                            'parse_obj_matmap=tls_occlusion.command_line:parse_obj_matmap',
                            'parse_obj_clone=tls_occlusion.command_line:parse_obj_clone',
                            'pc_correction=tls_occlusion.command_line:pc_correction',
                            'batch_pc_correction=tls_occlusion.command_line:batch_pc_correction',
                            'match_pf=tls_occlusion.command_line:match_pf',
                            'batch_match_pf=tls_occlusion.command_line:batch_match_pf']},
    url='https://github.com/mattbv/tls_occlusion',
    license='LICENSE.txt',
    description='Performs the matching of simulated point clouds to tree\
 tree models in order to assess scanning occlusion.',
    long_description=readme(),
    classifiers=['Programming Language :: Python',
                 'Topic :: Scientific/Engineering'],
    keywords='TLS occlusion point cloud LiDAR',
    install_requires=required,
    # ...
)
