[![build](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml/badge.svg)](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml)
[![Documentation Status](https://readthedocs.org/projects/sipkit/badge/?version=latest)](https://sipkit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sipkit.svg)](https://badge.fury.io/py/sipkit)

# Silicon Photonics Toolkit

## Getting Started

**Silicon Photonics Toolkit** is a toolkit providing fundamental waveguide and material properties to aid in the design of silicon photonic components on SOI platforms with high accuracy and extremely fast runtime. See the [documentation](https://sipkit.readthedocs.io/en/latest/) for tutorials and API reference.

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
    pip install -e .

## Dependencies

The package requires the following packages to be installed:

-   [NumPy](https://numpy.org/)
-   [Jax](https://jax.readthedocs.io/en/latest/index.html)

## Citing SiPhotonics Toolkit

    @software{silicon-photonics-toolkit2022github,
        url = {https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit},
        author = {Aycan Deniz Vit, Emir Salih Mağden},
        title = {Silicon Photonics Toolkit},
        description = {A toolkit to rapidly lookup parameters for the design of silicon photonic components with automatic differentiation capability.},
        year = {2022},
    }
