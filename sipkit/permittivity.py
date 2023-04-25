from __future__ import annotations
import os
import pickle
import jaxlib

from jax import jit
from jax.scipy.ndimage import map_coordinates


user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open("permittivity.pkl", "rb") as handle:
    perm = pickle.load(handle)
os.chdir(user_dir)

min_wav = perm['wavelengths'].min()
max_wav = perm['wavelengths'].max()
wav_size = len(perm['wavelengths'])

@jit
def perm_si(wavelength: float | list[float]) -> jaxlib.xla_extension.DeviceArray | jaxlib.xla_extension.Array:
    """
    Permittivity value of Si at given wavelength.
    
    Args:
        wavelength (float): Wavelength in microns. (1.2 - 1.7) float or list

    Returns:
        Permittivity value(s).

    Examples:
        >>> perm_si(1.5)
        11.68

        >>> perm_si([1.5, 1.6])
        [11.68, 11.68]
    """
    return map_coordinates(
        perm['Si'],
        [
            (wavelength - min_wav) * ((wav_size - 1) / (max_wav - min_wav)),
        ],
        order=1,
    )


@jit
def perm_oxide(wavelength: float | list[float]) -> jaxlib.xla_extension.DeviceArray | jaxlib.xla_extension.Array:
    """
    Permittivity value of SiO2 at given wavelength.

    Args:
        wavelength (float): Wavelength in microns. (1.2 - 1.7) float or list

    Returns:
        Permittivity value(s).

    Examples:
        >>> perm_oxide(1.5)
        3.44

        >>> perm_oxide([1.5, 1.6])
        [3.44, 3.44]
        
    """
    return map_coordinates(
        perm['SiO2'],
        [
            (wavelength - min_wav) * ((wav_size - 1) / (max_wav - min_wav)),
        ],
        order=1,
    )
