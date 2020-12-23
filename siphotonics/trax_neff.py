import os
import pickle
from trax import fastmath
from jax import jit
from jax.scipy import ndimage

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))

with open(r"trax_neff_array.pickle", "rb") as input_file:
    trax_neff_array = pickle.load(input_file)
os.chdir(user_dir)


@jit
def fast_neff(width, wl, mode):
    """
    Gets Effective Index value by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.25 - 0.7)
    :param wl: Wavelength in microns. (1.2 - 1.7)
    :param mode: Mode number. (1 - 5)
    :return: Effective Index value. Scalar
    """
    return ndimage.map_coordinates(trax_neff_array, [(width - 0.25) * 1000, (wl - 1.2) * 1000, mode - 1], order=1)


@jit
def grad_neff(width, wl, mode):
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
    return fastmath.grad(fast_neff, (0, 1))(width, wl, mode.astype(float))
