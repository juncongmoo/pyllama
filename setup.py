# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))


def _get_version():
    with open(os.path.join(here, "llama", "version.py")) as f:
        try:
            version_line = next(line for line in f if line.startswith("__version__"))
        except StopIteration:
            raise ValueError("__version__ not defined in itree/version.py")
        else:
            ns = {}
            exec(version_line, ns)  # pylint: disable=exec-used
            return ns["__version__"]


VERSION = _get_version()


def read_file(filename: str):
    try:
        lines = []
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith("#")]
        return lines
    except:
        return []


DESCRIPTION = "🦙 LLaMA: Open and Efficient Foundation Language Models in A Single GPU"

r_quant = read_file(f"{here}/requirements-quant.txt")
r_basic = read_file(f"{here}/requirements.txt")


def package_files(ds):
    paths = []
    for d in ds:
        for path, directories, filenames in os.walk(d):
            for filename in filenames:
                if "__pycache__" not in str(filename) and not filename.endswith('.pyc'):
                    paths.append(str(os.path.join(path, filename))[len("llama/") :])
    return paths


extra_files = package_files(["llama/"])

setup(
    name="pyllama",
    version=VERSION,
    author="Juncong Moo;Meta AI",
    author_email="JuncongMoo@gmail.com",
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=r_basic,
    package_data={"llama": extra_files},
    include_package_data=True,
    keywords=[
        "LLaMA",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/juncongmoo/pyllama",
    packages=["llama"],
    extras_require={
        "quant": r_quant,
        "full": r_quant + r_basic,
    },
)
