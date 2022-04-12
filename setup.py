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
    version='0.1',
    description='Plug-in-and-Play Toolbox for Stablizing Transformer Training',
    long_description= read_readme(),
    long_description_content_type="text/markdown",
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='Apache License 2.0',
    entry_points={
        'console_scripts': ['torch_scope=torch_scope.commands:run'],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*