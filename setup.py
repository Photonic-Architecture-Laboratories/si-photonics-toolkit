from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

setup(
    name="siphotonics",
    version="1.1",
    description="Silicon Photonics Development Package",
    long_description="...",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="silicon photonics optics waveguide",
    url="https://github.com/aycandv/siphotonics",
    author="Aycan Deniz Vit",
    author_email="avit16@ku.edu.tr",
    packages=["siphotonics"],
    install_requires=[
        'numpy',
        'h5py',
        'scipy',
    ],
    #scripts=['bin/funniest-joke'],
    include_package_data=True,
    zip_safe=False)
