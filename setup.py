"""DETR setup.

Copyright 2020
"""
from setuptools import setup, find_packages
import os

dirname = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dirname, 'README.md')) as f:
    long_description = f.read()


setup(
    name='detr',
    version='0.0.1',
    description="DETR implementation.",
    long_description=long_description,
    url='https://github.com/PhilJd/detr',

    author='Facebook Research',
    author_email='',

    keywords='pytorch DETR',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'torchvision'],

    license='Apache 2.0',

    classifiers=[
        # Change later on:
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
