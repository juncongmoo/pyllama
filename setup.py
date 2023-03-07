# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))

def read_file(filename: str):
    try:
        lines = []
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith('#')]
        return lines
    except:
        return []

DESCRIPTION = "🦙 LLaMA: Open and Efficient Foundation Language Models in A Single GPU"

setup(name="pyllama",
    version="0.0.2",
    author="Juncong Moo;Meta AI",
    author_email="JuncongMoo@gmail.com",
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=read_file(f"{here}/requirements.txt"),
    keywords=[
        "LLaMA",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/juncongmoo/pyllama",
    packages=find_packages()
    )


