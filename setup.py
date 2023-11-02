#/usr/bin/env python3
import setuptools

with open("README.md", "r") as fh:
    long_description=fh.read()

setuptools.setup(
        name="handwriting_datasets",
        python_requires='>=3.8',
        version="0.1",
        author="Nicolas Renet",
        author_email="nprenet@gmail.com",
        description="Torch dataset classes",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=["handwriting_datasets"],
)
