# siphotonics

**siphotonics** is a package to provide fundamental parameters to aid in the design of silicon photonic components on an SOI platform with high accuracy in extremely fast runtime.

* There are sample codes for the package in a folder which is named as **Example Notebooks** in github repository and Google Drive folder.

# Guide for Local Installation

Google Colab Users (for Laboratory)
-----------------------------------
If you want to work with Colab, you only need to insert the code fragment below in your notebook to use **siphotonics**:

```python
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/Shared drives/PAL Drive/Software/SiPhotonics Python/siphotonics v0.1')
!pip install .
```
* Notice *siphotonics v0.1* at the end of the ``chdir`` command. If you encounter any errors, go to the *SiPhotonics Python* directory and check the latest version number. Other/external users will need to modify the directory name and path depending on where they place the siphotonics package.

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
    >>> siphotonics.neff(500,  1550, "te1")
    >>> siphotonics.neff(500,  1550, "tM0")
    >>> siphotonics.neff(500,  1550, "TE1")
    >>> siphotonics.neff(500,  1550,     1)
    >>> siphotonics.neff(500,  1550,     2)
                          ^      ^      ^
                       width  wavelng  mode

Polarization Fraction of First Five Modes
---------------------------------------------
    >>> siphotonics.polarization_frac(500,  1550, "tm1")
    >>> siphotonics.polarization_frac(500,  1550, "tE0")
    >>> siphotonics.polarization_frac(500,  1550, "Te1")
    >>> siphotonics.polarization_frac(500,  1550,     1)
    >>> siphotonics.polarization_frac(500,  1550,     5)
                                       ^      ^      ^
                                    width  wavelng  mode

Permittivity of Si & SiO2
-----------------------------
    >>> siphotonics.perm_si(1550)
    >>> siphotonics.perm_oxide(1550)
                                ^
                            wavelength
