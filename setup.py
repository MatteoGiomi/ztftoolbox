#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# setup script for ztftools
#
# Author: M. Giomi (matteo.giomi@desy.de)

from setuptools import setup
setup(
    name='ztftoolbox',
    version='0.1',
    description='make ZTF analysis simple again',
    author='Matteo Giomi',
    author_email='matteo.giomi@desy.de',
    packages=['ztftoolbox'],
    url = 'https://github.com/MatteoGiomi/ztftoolbox',
    install_requires=['astropy', 'pandas', 'sqlalchemy', 'psycopg2']
    )
