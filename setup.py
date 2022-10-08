# -*- coding: utf-8 -*-
# Copyright (C) 2022 Jarry Chen
# Licence: MIT

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


with open('README.md', encoding="utf-8") as f:
    long_description = f.read()


def setup_package():
    setup(
        name='dag_gen',
        version='0.0.1',
        description='A causal graph generator',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=['examples', 'tests', 'tests.*']),
        url='https://github.com/JayJayc/dag_gen',
        package_data={'': ['**/*.R', '**/*.csv']},
        install_requires=['numpy', 'scipy', 'pandas',
                          'networkx', 'skrebate'],
        include_package_data=True,
        author='Jarry Chen',
        author_email='jarrry.chen@gmail.com',
        license='MIT License')


if __name__ == '__main__':
    setup_package()
