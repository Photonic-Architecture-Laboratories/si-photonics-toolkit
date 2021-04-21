import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pickle
import h5py
import jax
from jax.scipy import ndimage
from jax import numpy as jnp

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open(r"coupling_coefficient_TE0-TE0.pickle", "rb") as input_file:
    couplingCoefficientArray = pickle.load(input_file)
dc_mesh_2 = h5py.File('dc-mesh-2.mat', "r")
dc_mesh_3 = h5py.File('dc-mesh-3.mat', "r")
dc_mesh_4 = h5py.File('dc-mesh-4.mat', "r")
wl_interval = jnp.linspace(1.2, 1.7, 101)
os.chdir(user_dir)

width1_array = np.linspace(300, 701, 41)
width2_array = np.linspace(300, 701, 41)
wavelengthArray = np.arange(1200, 1701, 10)
gapArr = np.logspace(np.log(100), np.log(2001), num=150, endpoint=True, base=np.e, dtype=int)
points = (width1_array, width2_array, gapArr, wavelengthArray)
my_interp = RegularGridInterpolator(points, couplingCoefficientArray)


def coupling_coefficient(width1, width2, gap, wavelength):
    """

    :param width1:
    :param width2:
    :param gap:
    :param wavelength:
    :return:
    """
    if gap >= 2:
        return 0.00001
    return my_interp([width2 * 1000, width1 * 1000, gap * 1000, wavelength * 1000])[0]
    pass


def through_power(wavelength, mesh=4):
    """

    :param wavelength:
    :param mesh:
    :return:
    """
    def get_file(_mesh):
        if _mesh == 2:
            return dc_mesh_2
        elif _mesh == 3:
            return dc_mesh_3
        elif _mesh == 4:
            return dc_mesh_4

    file = get_file(mesh)
    x0 = jnp.array(file["lum"]["x0"])[::-1]
    t_square = jnp.array(file["lum"]["y0"])[::-1]
    interpolated_t_sq = jnp.interp(wl_interval, x0.reshape(len(x0)) * 1e6, t_square.reshape(len(t_square)))
    return jax.scipy.ndimage.map_coordinates(interpolated_t_sq, [(wavelength - 1.2) * (100 / 0.5)], order=1)
