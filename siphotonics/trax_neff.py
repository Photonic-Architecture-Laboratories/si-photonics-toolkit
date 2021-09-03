import os
import h5py
import numpy as np
from trax import fastmath
from jax import jit
from jax.scipy import ndimage
from jax import numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))

with open('neff_width_240_20_700_wav_1200_0p1_1700.csv') as file:
    lines = file.readlines()

os.chdir(user_dir)

neff_data = jnp.array(list(map(float, lines[1][:-2].split(','))))
width_data = jnp.array(list(map(float, lines[3][:-2].split(','))))
wav_data = jnp.array(list(map(float, lines[5][:-2].split(','))))

wav_size = wav_data.shape[0]
wav_min = np.min(wav_data)
wav_max = np.max(wav_data)

width_size = width_data.shape[0]
width_min = np.min(width_data)
width_max = np.max(width_data)

neff_data = jnp.reshape(neff_data, (wav_size, width_size))


@jit
def neff(width, wavelength, mode=1):
    """
    Gets Effective Index value by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.25 - 0.7) Scalar or list
    :param wavelength:
    :param mode: Mode number. (1 - 5)
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(neff_data,
                                   [(wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
                                    (width - width_min) * ((width_size - 1) / (width_max - width_min))],
                                   order=1)


@jit
def grad_neff(width, wl, mode=1):
    """
    Gets derivatives of Effective Index with respect to waveguide width and
    wavelength.
    :param width: Waveguide width in microns. (0.25 - 0.7)
    :param wl: Wavelength in microns. (1.2 - 1.7)
    :param mode: Mode number. (1 - 5)
    :return: A tuple of DeviceArrays. First element corresponds to derivative
    with respect to width. Last element is the derivative with respect to
    wavelength
    """
    return fastmath.grad(neff, (0, 1))(width, wl, mode.astype(float))
