# siphotonics

**siphotonics** is a package that provides fundamental waveguide and material parameters to aid in the design of silicon photonic components on SOI platforms with high accuracy and extremely fast runtime.

* There are sample codes for the package in the **Example Notebooks** folder.

# Guide for Local Installation

Google Colab Users (for Laboratory)
-----------------------------------
If you want to work with Colab, you only need to insert the code fragment below in your notebook to use **siphotonics**:

```python
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/Shared drives/PAL Drive/Software/SiPhotonics Python/siphotonics')
import siphotonics as sip
```

Windows Users
-------------

* Make sure you download Python with adding it to PATH at download step.
* Control if "pip" is installed or not by typing ``pip -V``.

    * If "pip" is not installed, download [pip.py](https://pypi.org/project/pip/), then type ``python get-pip.py``.
    
* Go to the directory where you clone this repo.
* Type ``pip install .``
* Done.

Linux Users
-----------
* Go to the directory where you clone this repo.
* Type ``pip install .``
* Done.

# Usage:


Effective Index Values of First Five Modes
----------------------------------------------
    >>> siphotonics.neff(0.5,  1.55, "te1")
    >>> siphotonics.neff(0.5,  1.55, "tM0")
    >>> siphotonics.neff(0.5,  1.55, "TE1")
    >>> siphotonics.neff(0.5,  1.55,     1)
    >>> siphotonics.neff(0.5,  1.55,     2)
                          ^      ^      ^
                       width  wavelng  mode

Group Index 
----------------------------------------------
    >>> siphotonics.ng(0.5,  1.55)
                        ^      ^
                     width  wavelength

Polarization Fraction of First Five Modes
---------------------------------------------
    >>> siphotonics.polarization_frac(0.5,  1.55, "tm1")
    >>> siphotonics.polarization_frac(0.5,  1.55, "tE0")
    >>> siphotonics.polarization_frac(0.5,  1.55, "Te1")
    >>> siphotonics.polarization_frac(0.5,  1.55,     1)
    >>> siphotonics.polarization_frac(0.5,  1.55,     5)
                                       ^      ^      ^
                                    width  wavelng  mode

Permittivity of Si & SiO2
-----------------------------
    >>> siphotonics.perm_si(1.55)
    >>> siphotonics.perm_oxide(1.55)
                                ^
                            wavelength

Coupling Coefficient Between Two Waveguides
-----------------------------
    >>> siphotonics.couplingCoefficient(0.5, 0.5, 0.2, 1.55)
                                        ^     ^    ^     ^ 
                                    width1 width2 gap  wavelength
