import os
import pickle

import numpy as np

from siphotonics.effective_index import neff

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open(r"Si_Palik_Refractive_Index.pickle", "rb") as input_file:
    refractive_index = pickle.load(input_file)
os.chdir(user_dir)

wavelength_array = np.linspace(1.2, 1.7, 101)
neff_array = []
for i in wavelength_array:
    neff_array.append(neff(0.5, i))

difference = np.diff(neff_array) / np.diff(wavelength_array)


def group_index(width, wavelength):
    """
    Group Index of light at a specified wavelength in a waveguide with a specified width.
    :param width:
    :param wavelength:
    :return:
    """
    if wavelength > 1.7 or wavelength < 1.2:
        raise ValueError("Wavelength must be between 1.2 and 1.7 microns.")
    if width > 0.7 or width < 0.3:
        raise ValueError("Width must be between 0.3 and 0.7 microns.")

    if wavelength == 1.7:
        n_g = neff(width, wavelength) - wavelength * (neff(width, wavelength) - neff(width, wavelength - 0.001)) / 0.001
    else:
        n_g = neff(width, wavelength) - wavelength * (neff(width, wavelength + 0.001) - neff(width, wavelength)) / 0.001
    return n_g
