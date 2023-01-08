from __future__ import annotations
import os
import pickle

from sipkit.effective_index import wav_max, wav_min

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open("permittivity.pickle", "rb") as handle:
    perm = pickle.load(handle)
os.chdir(user_dir)


def perm_si(wavelength: float | list[float]) -> float | list[float]:
    """
    Permittivity value of Si at given wavelength.
    
    Args:
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Permittivity value(s).

    Raises:
        ValueError: If wavelength is not between 1.2-1.7 microns.

    Examples:
        >>> perm_si(1.5)
        11.68

        >>> perm_si([1.5, 1.6])
        [11.68, 11.68]

        >>> perm_si(1.8)
        Traceback (most recent call last):
            ...
        ValueError: Wavelength must be between 1.2-1.7 micron
    """
    if not wav_min <= wavelength <= wav_max:
        raise ValueError("Wavelength must be between 1.2-1.7 micron")

    return perm["Si"](wavelength * 1000)


def perm_oxide(wavelength: float | list[float]) -> float | list[float]:
    """
    Permittivity value of SiO2 at given wavelength.

    Args:
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Permittivity value(s).

    Raises:
        ValueError: If wavelength is not between 1.2-1.7 microns.

    Examples:

        >>> perm_oxide(1.5)
        3.44

        >>> perm_oxide([1.5, 1.6])
        [3.44, 3.44]

        >>> perm_oxide(1.8)
        Traceback (most recent call last):
            ...
        ValueError: Wavelength must be between 1.2-1.7 micron
        
    """
    if not wav_min <= wavelength <= wav_max:
        raise ValueError("Wavelength must be between 1.2-1.7 micron")

    return perm["SiO2"](wavelength * 1000)
