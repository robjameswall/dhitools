# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='dhitools',
    version='0.0.4',
    description='Python tools for working with DHI MIKE21',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Rob Wall',
    author_email='rob.james.wall@gmail.com',
    url='https://github.com/robjameswall/dhitools',
    license='LICENSE.txt',
    packages=['dhitools'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=[
        "numpy == 1.14.5",
        "pythonnet==2.3.0",
        "PyCRS==0.1.3",
        "scipy==1.1.0"
    ],
    extras_requires={
        'gis': ["GDAL==2.2.4",
                "geopandas==0.3.0"]
    }
)