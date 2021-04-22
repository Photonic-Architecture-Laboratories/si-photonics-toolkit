import siphotonics as sip
import numpy as np


def fsr(wavelength, radius, width):
    """
    Free Spectral Range
    :param wavelength:
    :param radius:
    :param width:
    :return:
    """
    return (wavelength ** 2) / (2 * np.pi * radius * sip.ng(width, wavelength))
