#!/usr/bin/env/python
"""
Installation script
"""


import os
from setuptools import find_packages, setup


LONG_DESCRIPTION = """ #TODO
Package description
"""

if os.environ.get("READTHEDOCS", False) == "True":
    INSTALL_REQUIRES = []
    EXTRAS_REQUIRES = []
else:
    INSTALL_REQUIRES = []

    EXTRAS_REQUIRES = {}


setup(
    name="src",
    version="0.1.0",
    description="Analysis of European Storm Petrel song",
    license="MIT",
    author='Nilo M. Recalde',
    url=".",
    long_description=LONG_DESCRIPTION,
    packages=["src"],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True
)
