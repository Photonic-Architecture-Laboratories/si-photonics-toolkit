[![build](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml/badge.svg)](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml)
[![Documentation Status](https://readthedocs.org/projects/sipkit/badge/?version=latest)](https://sipkit.readthedocs.io/en/latest/?badge=latest)


# SiPhotonics Toolkit

## Getting Started

**SiPhotonics Toolkit** is a toolkit providing fundamental waveguide and material properties to aid in the design of silicon photonic components on SOI platforms with high accuracy and extremely fast runtime. See the [documentation](https://sipkit.readthedocs.io/en/latest/) for tutorials and API reference.

## Installation

### Pip

The package can be installed via pip:

    pip install sipkit

You can install siphotonics with additional packages for developers:

    pip install sipkit[dev]

### Build from source

Alternatively, the package can be built from source by cloning the repository and running the setup script:

    git clone https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit.git
    cd si-photonics-toolkit
    python setup.py install

## Dependencies

The package requires the following packages to be installed:

-   [NumPy](https://numpy.org/)
-   [Jax](https://jax.readthedocs.io/en/latest/index.html)
-   [Trax](https://trax-ml.readthedocs.io/en/latest/)

## Citing SiPhotonics Toolkit

    @software{si-photonics-toolkit2022github,
        url = {https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit},
        author = {Aycan Deniz Vit, Emir Salih Mağden},
        title = {SiPhotonics Toolkit: A toolkit for the design of silicon photonic components},
        year = {2022},
    }
