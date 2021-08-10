import os
import h5py
import numpy as np
from trax import fastmath
from jax import jit
from jax.scipy import ndimage
from jax import numpy as jnp

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))

f_neff = h5py.File("neff.mat", "r")
os.chdir(user_dir)

width_span = jnp.array(f_neff["width_sp"]) * 1e6
wavelength_span = jnp.array(f_neff["wavelength_span"]) * 1e6
eff_ind = np.array(f_neff["neff"], dtype=np.complex128).real


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
    return ndimage.map_coordinates(eff_ind,
                                   [(wavelength - 1.2) * (25 / 0.5), (width - 0.24) * (23 / (0.7 - 0.24))],
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
