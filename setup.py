from setuptools import find_packages, setup


def read_version(fname="generalization/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


def parse_requires_deps():
    requirements = []
    dependency_links = []

    with open("requirements.txt", "r") as f:
        for line in f:
            if line.startswith("-f"):
                dependency_links.append(line.split(" ")[1].strip())
            else:
                requirements.append(line.strip())

    return requirements, dependency_links


requirements, dependency_links = parse_requires_deps()

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
    url="https://github.com/ids-uchile/Generalization",
    license="BSD-3-Clause",
    packages=find_packages(exclude=["tests*"]),
    # install_requires=requirements,
    # dependency_links=dependency_links,
    include_package_data=True,
    extras_require={"dev": ["pytest", "black", "flake8", "isort"]},
)
