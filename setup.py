#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="probpipe",                                 # your PyPI/distribution name
    version="0.1.0",
    description="Tools for probabilistic data‐processing pipelines",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Yongho Lim",
    author_email="ylim2@bu.edu",
    url="https://github.com/TARPS-group/prob-pipe",

    # this will find your probpipe/ package (and any subpackages),
    # but exclude tests, docs, notebooks, etc.
    packages=find_packages(exclude=["tests*", "docs*", "notebooks*"]),

    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "nbsphinx",
            "black",
            "flake8",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    include_package_data=False,   # True, if you have a MANIFEST.in or package_data
)