from setuptools import setup

PACKAGE = "CNClassifier"
NAME = "CNClassifier"
DESCRIPTION = "This package aims to do text classification of clinical notes."
AUTHOR = "Wei Ruan"
AUTHOR_EMAIL = "acdmc.wruan@gmail.com"
URL = "https://github.com/patrick013"
VERSION = "1.0.4"
with open("README.md", "r") as fh:
    Long_Description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=Long_Description,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="Apache License, Version 2.0",
    url=URL,
    packages=["CNClassifier"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)

