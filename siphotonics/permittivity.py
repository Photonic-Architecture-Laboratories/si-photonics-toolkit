import os
import pickle

from siphotonics.effective_index import wav_min, wav_max

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open("permittivity.pickle", "rb") as handle:
    perm = pickle.load(handle)
os.chdir(user_dir)


def perm_si(wavelength):
    """
    Permittivity value of Si at given wavelength.
    :param wavelength:
    :return:
    """
    if not wav_min <= wavelength <= wav_max:
        raise ValueError("Wavelength must be between 1.2-1.7 micron")

    return perm["Si"](wavelength * 1000)


def perm_oxide(wavelength):
    """
    Permittivity value of SiO2 at given wavelength.
    :param wavelength:
    :return:
    """
    if not wav_min <= wavelength <= wav_max:
        raise ValueError("Wavelength must be between 1.2-1.7 micron")

    return perm["SiO2"](wavelength * 1000)
