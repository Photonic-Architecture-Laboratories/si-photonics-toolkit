# siphotonics

**siphotonics** is a package to provide fundamental features of a SOI platform with high accuracy and small running time.

# Guide for Local Installation

Google Colab Users (for Laboratory)
-----------------------------------
If you want to work with Colab, you only need to insert the code fragment below in your notebook to use **siphotonics**:

```python
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/Shared drives/PAL Drive/Software/SiPhotonicsPython/siphotonics')
!pip install .
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
