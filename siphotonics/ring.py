import numpy as np

from siphotonics.group_index import group_index


def fsr(wavelength, radius, width):
    """
    Free Spectral Range
    :param wavelength:
    :param radius:
    :param width:
    :return:
    """
    return (wavelength ** 2) / (2 * np.pi * radius * group_index(width, wavelength))
