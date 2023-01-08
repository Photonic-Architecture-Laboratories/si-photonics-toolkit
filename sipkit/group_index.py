from __future__ import annotations
from sipkit.effective_index import neff, wav_max, wav_min, width_max, width_min


def group_index(width, wavelength):
    """
    Group Index of light at a specified wavelength in a waveguide with a specified width.
    :param width:
    :param wavelength:
    :return:
    """
    if wavelength > wav_max or wavelength < wav_min:
        raise ValueError("Wavelength must be between 1.2 and 1.7 microns.")
    if width > width_max or width < width_min:
        raise ValueError("Width must be between 0.3 and 0.7 microns.")

    if wavelength == wav_max:
        n_g = neff(width, wavelength) - wavelength * (neff(width, wavelength) - neff(width, wavelength - 0.001)) / 0.001
    else:
        n_g = neff(width, wavelength) - wavelength * (neff(width, wavelength + 0.001) - neff(width, wavelength)) / 0.001
    return n_g
