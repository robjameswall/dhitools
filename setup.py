# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='dhitools',
    version='0.1.0',
    description='Python tools for working with DHI MIKE21',
    long_description=open('README.md').read(),
    author='Rob Wall',
    url='https://github.com/robjameswall',
    license='LICENSE.txt',
    packages=['dhitools'],
    install_requires=[
        "numpy == 1.14.5",
        "pythonnet==2.3.0",
        "python-dotenv==0.8.2",
        "PyCRS==0.1.3"
    ],
    data_files=['.env'],
    extras_requires={
        'gis': ["scipy==1.1.0",
                "GDAL==2.2.4",
                "geopandas==0.3.0"]
    }
)