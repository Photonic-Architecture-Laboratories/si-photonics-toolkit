from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests", "tests.*", "site", "site.*", "docs", "docs.*"])
)
