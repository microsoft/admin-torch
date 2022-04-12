#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def read_readme():
    with open('README.md') as f:
        return f.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'torch'
]

setup(
    name='admin_torch',
    version='0.1.0',
    description='Plug-in-and-Play Toolbox for Stablizing Transformer Training',
    long_description= read_readme(),
    long_description_content_type="text/markdown",
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='https://github.com/microsoft/admin-torch',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='Apache License 2.0',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*