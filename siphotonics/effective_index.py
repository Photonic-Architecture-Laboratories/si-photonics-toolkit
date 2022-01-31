from jax import jit
from jax.scipy import ndimage
from trax import fastmath

from siphotonics.read_data import (
    effective_index_te0,
    effective_index_te1,
    effective_index_te2,
    effective_index_tm0,
    effective_index_tm1,
    neff_data,
    wav_max,
    wav_min,
    wav_size,
    width_max,
    width_min,
    width_size,
)


@jit
def neff(width, wavelength):
    """
    Gets Effective Index value by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.25 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        neff_data,
        [
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
        ],
        order=1,
    )


@jit
def neff_te0(width, wavelength):
    """
    Gets Effective Index value  of TE0 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.24 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_te0,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )


@jit
def neff_tm0(width, wavelength):
    """
    Gets Effective Index value  of TM0 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.24 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_tm0,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )


@jit
def neff_te1(width, wavelength):
    """
    Gets Effective Index value  of TE1 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.24 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_te1,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )


@jit
def neff_tm1(width, wavelength):
    """
    Gets Effective Index value  of TM1 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.24 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_tm1,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )


@jit
def neff_te2(width, wavelength):
    """
    Gets Effective Index value  of TE2 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.
    :param width: Waveguide width in microns. (0.24 - 0.7) Scalar or list
    :param wavelength:
    :return: Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_te2,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )


@jit
def grad_neff(width, wavelength, mode=1):
    """
    Gets derivatives of Effective Index with respect to waveguide width and
    wavelength.
    :param width: Waveguide width in microns. (0.25 - 0.7)
    :param wavelength: Wavelength in microns. (1.2 - 1.7)
    :param mode: Mode number. (1 - 5)
    :return: A tuple of DeviceArrays. First element corresponds to derivative
    with respect to width. Last element is the derivative with respect to
    wavelength
    """
    return fastmath.grad(neff, (0, 1))(width, wavelength)
