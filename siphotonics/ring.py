import numpy as np

import siphotonics as sip


def fsr(wavelength, radius, width):
    """
    Free Spectral Range
    :param wavelength:
    :param radius:
    :param width:
    :return:
    """
    return (wavelength ** 2) / (2 * np.pi * radius * sip.group_index(width, wavelength))
