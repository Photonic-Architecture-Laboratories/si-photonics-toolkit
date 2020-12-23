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

    :param width:
    :param wl:
    :param mode:
    :return:
    """
    return ndimage.map_coordinates(trax_neff_array, [(width - 0.25) * 1000, (wl - 1.2) * 1000, mode - 1], order=1)


@jit
def grad_neff(width, wl, mode):
    """
    
    :param width:
    :param wl:
    :param mode:
    :return:
    """
    return fastmath.grad(fast_neff, (0, 1))(width, wl, mode.astype(float))
