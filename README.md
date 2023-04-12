[![build](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml/badge.svg)](https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit/actions/workflows/makefile.yml)
[![Documentation Status](https://readthedocs.org/projects/sipkit/badge/?version=latest)](https://sipkit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sipkit.svg)](https://badge.fury.io/py/sipkit)

# Silicon Photonics Toolkit

Various software packages and resources for rapid parameter lookup and calculation have been developed across a range of disciplines [1-17]. These resources offer easy and efficient access to standard parameters used in related areas of research. However, numerical simulators and equation solvers are often computationally expensive tools that require significant resources for simple data retrieval and calculations. To address this challenge, we present Silicon Photonics Toolkit that offers auto-differentiation and fast data retreival capabilities. The pre-saved data has been generated using Lumerical MODE Solutions' FDE Solver and subsequently interpolated and mapped using a state-of-the-art Python package JAX. Data mapping function from JAX makes data lookup operations achieved in extremely fast running time, making it a valuable resource for researchers in the silicon photonics field. Silicon Photonics Toolkit provides a user-friendly and easy to use interface that enables rapid data access, leading to time-efficient research and accelerated progress in scientific discovery.

## Getting Started

**Silicon Photonics Toolkit** is a toolkit providing fundamental waveguide and material properties to aid in the design of silicon photonic components on silicon-on-insulator platforms with high accuracy and extremely fast runtime. All the waveguide parameters returned by sipkit are calculated for 220-nm-thick strip waveguides on a SOI. See the [documentation](https://sipkit.readthedocs.io/en/latest/) for tutorials and API reference.

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
        author = {Aycan Deniz Vit and Kazım Görgülü and Ali Najjar Amiri and Emir Salih Mağden},
        title = {Silicon Photonics Toolkit},
        description = {A toolkit to rapidly lookup parameters for the design of silicon photonic components with automatic differentiation capability.},
        year = {2022},
    }
    
## References

1- Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. Benchmarking Materials Property
Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm. npj 
Computational Materials 6, 138 (2020). https://doi.org/10.1038/s41524-020-00406-3

2- Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028

3- A.S. Rosen, S.M. Iyer, D. Ray, Z. Yao, A. Aspuru-Guzik, L. Gagliardi, J.M. Notestein, R.Q. Snurr. "Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery", Matter, 4, 1578-1597 (2021). DOI: 10.1016/j.matt.2021.02.015.

4- A.S. Rosen, V. Fung, P. Huck, C.T. O'Donnell, M.K. Horton, D.G. Truhlar, K.A. Persson, J.M. Notestein, R.Q. Snurr. "High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration," npj Comput. Mat., 8, 112 (2022). DOI: 10.1038/s41524-022-00796-6.

5- Lu, Lu, et al. "Extraction of mechanical properties of materials through deep learning from instrumented indentation." Proceedings of the National Academy of Sciences 117.13 (2020): 7052-7062.

6- Caleb Bell and Contributors (2016-2021). Thermo: Chemical properties component of Chemical Engineering Design Library (ChEDL) https://github.com/CalebBell/thermo

7- Caleb Bell (2016-2023). fluids: Fluid dynamics component of Chemical Engineering Design Library (ChEDL) https://github.com/CalebBell/fluids

8- https://github.com/wigging/chemics

9- https://github.com/dukenmarga/civil-engineering-toolbox

10- https://github.com/hkaneko1985/dcekit

11- https://github.com/jtambasco/opticalmaterialspy

12- https://github.com/RasaHQ/rasa_lookup_demo

13- https://github.com/berenslab/EphysExtraction

14- https://github.com/CitrineInformatics/MPEA_dataset

15- https://refractiveindex.info/

16- https://pppdb.uchicago.edu/

17- https://oqmd.org/
