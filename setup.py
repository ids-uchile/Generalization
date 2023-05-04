import os

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="generalization/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []

setup(
    name="ids-generalization",
    py_modules=["generalization"],
    version=read_version(),
    description="Experiment Suite for Understanding and Rethinking Generalization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Stepp1",
    url="https://github.com/ds-uchile/Rethinking-Generalization",
    license="BSD-3-Clause",
    # packages=find_packages(exclude=["tests*"]),
    packages=find_packages(),
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest", "black", "flake8", "isort"]},
)
