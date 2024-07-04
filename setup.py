#/usr/bin/env python3
import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description=fh.read()

setuptools.setup(
        name="handwriting_datasets",
        version="0.1",

        author="Nicolas Renet",
        author_email="nprenet@gmail.com",
        description="Torch dataset classes",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages()
)
