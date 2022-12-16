# Getting Started

**SiPhotonics** is a package providing fundamental waveguide and material properties to aid in the design of silicon photonic components on SOI platforms with high accuracy and extremely fast runtime.

## Installation

### Pip

The package can be installed via pip:

    pip install siphotonics

### Build from source

Alternatively, the package can be built from source by cloning the repository and running the setup script:

    git clone https://github.com/Photonic-Architecture-Laboratories/siphotonics.git
    cd siphotonics
    python setup.py install

## Dependencies

The package requires the following packages to be installed:

-   [NumPy](https://numpy.org/)
-   [SciPy](https://www.scipy.org/)
-   [Matplotlib](https://matplotlib.org/)
-   [Jax](https://jax.readthedocs.io/en/latest/index.html)
-   [Trax](https://trax-ml.readthedocs.io/en/latest/)

## BibTeX

    @software{siphotonics2022github,
        url = {https://github.com/Photonic-Architecture-Laboratories/siphotonics},
        author = {Aycan Deniz Vit, Emir Salih Mağden},
        title = {SiPhotonics: A Python package for the design of silicon photonic components},
        year = {2022},  
        version = {0.1.0},
    }
