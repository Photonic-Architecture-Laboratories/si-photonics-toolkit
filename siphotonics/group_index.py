import os
from scipy import constants
import siphotonics as sip
import numpy as np
import pickle

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open(r"Si_Palik_Refractive_Index.pickle", "rb") as input_file:
    refractive_index = pickle.load(input_file)
os.chdir(user_dir)

wavelength_array = np.linspace(1.2, 1.7, 101)
neff_array = []
for i in wavelength_array:
    neff_array.append(sip.neff(0.5, i, 1))

difference = np.diff(neff_array)/np.diff(wavelength_array)

def ng(width, wavelength):
    if wavelength > 1.7 or wavelength < 1.2:
        raise ValueError("Wavelength must be between 1.2 and 1.7 microns.")
    if width > 0.7 or width < 0.3:
        raise ValueError("Width must be between 0.3 and 0.7 microns.")

    if wavelength == 1.7:
        n_g = sip.neff(width, wavelength,1) - wavelength * (sip.neff(width, wavelength,1) - sip.neff(width, wavelength-0.001,1))/(0.001)
        #n_g = sip.neff(width, wavelength,1) - wavelength * difference[int((wavelength-1.2)*1000/5)]
    else:
        n_g = sip.neff(width, wavelength,1) - wavelength * (sip.neff(width, wavelength+0.001,1) - sip.neff(width, wavelength,1))/(0.001)
        #n_g = sip.neff(width, wavelength,1) - wavelength * difference[int((wavelength-1.2)*1000/5)]
    return n_g

"""
print(ng(0.5, 1.55))

group_array = []

wavelengthArr = np.linspace(1.2, 1.69, 501)
for i in wavelengthArr:
    group_array.append(ng(0.5,i))

import matplotlib.pyplot as plt 

plt.plot(wavelengthArr, group_array)
plt.show()
"""