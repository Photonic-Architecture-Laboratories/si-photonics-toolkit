from setuptools import find_packages, setup


def readme():
    """
    Read README.md
    :return:
    """
    with open("README.md") as file:
        return file.read()


setup(
    name="siphotonics",
    version="0.7",
    description="Silicon Photonics Development Package",
    long_description=readme(),
    entry_points={
        "console_scripts": ["siphotonics=siphotonics.command_line:main"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="silicon photonics optics waveguide",
    url="",
    author="Aycan Deniz Vit",
    author_email="avit16@ku.edu.tr",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "trax",
        "pylint",
        "pytest",
        "click",
        "black",
        "pytest-cov",
    ],
    include_package_data=True,
    zip_safe=False,
)
