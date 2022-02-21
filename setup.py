#!/usr/bin/env python3
import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='minigrad',
        version='0.0.1',
        description='Toy autodiff library.',
        author='Marko Elez',
        license='MIT',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages = ['minigrad'],
        install_requires=['numpy'],
        python_requires='>=3.9',
        extras_require={
            'testing': [
                "pytest",
                "torch",
                "tqdm",
                ],
            },
        include_package_data=True)
