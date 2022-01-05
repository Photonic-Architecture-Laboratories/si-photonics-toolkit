# siphotonics

**siphotonics** is a package that provides fundamental waveguide and material parameters to aid in the design of silicon photonic components on SOI platforms with high accuracy and extremely fast runtime.

* For tutorials, see **Example Notebooks** folder.

## Installation

### Colab
If you use Colab environment, you need to import **siphotonics** using the code snippet below:

```python
!pip install -q -U git+https://github.com/google/trax@master

from google.colab import drive
import os

root = os.getcwd()
drive.mount('/content/gdrive')
os.chdir('/content/gdrive/Shared drives/PAL Drive/Software/SiPhotonics Python/siphotonics')

import siphotonics as sip
os.chdir(root)
```

### Windows, Linux and macOS
* Go to the directory where you clone this repo.
* Execute `pip install .`
* Done.

## Usage

### Effective Index
```python
sip.neff(width=0.5,  wavelength=1.55)
sip.neff(width=0.5,  wavelength=1.55)
```

### Group Index
```python
sip.ng(width=0.5,  wavelength=1.55)
```

### Polarization Fraction of First Five Modes
```python
sip.polarization_frac(width=0.5,  wavelength=1.55, mode="tm1")
sip.polarization_frac(width=0.5,  wavelength=1.55, mode="tE0")
sip.polarization_frac(width=0.5,  wavelength=1.55, mode="Te1")
sip.polarization_frac(width=0.5,  wavelength=1.55, mode=1)
sip.polarization_frac(width=0.5,  wavelength=1.55, mode=5)
```

### Permittivity of Si & SiO2
```python
sip.perm_si(wavelength=1.55)
sip.perm_oxide(wavelength=1.55)
```

### Derivative of an Array
```python
sip.derivative(data, order, step)
```
    
* **data**: Array-like data structure
* **order**: Order of derivative. Can be an integer from 1 to 5. By default, 1.
* **step**: Step size between indices of "data". For example, step size of the array `np.linspace(0, 2, 100)` equals 2/100. 
