from __future__ import annotations
from jax import jit
from jax.scipy import ndimage


from sipkit.read_data import (
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
def neff(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).

    Examples:
        >>> neff(0.5, 1.5)
        array(2.)

        >>> neff(0.5, [1.5, 1.6])
        array([2. , 2.1])

        >>> neff([0.5, 0.6], 1.5)
        array([2. , 1.9])

        >>> neff([0.5, 0.6], [1.5, 1.6])
        array([[2. , 2.1],

        >>> neff(0.5, [[1.5, 1.6], [1.7, 1.8]])
        array([[2. , 2.1],
               [1.9, 2. ]])

        >>> neff([[0.5, 0.6], [0.7, 0.8]], 1.5)
        array([[2. , 1.9],
               [1.8, 1.7]])

        >>> waveguide_width = jnp.array([0.5, 0.6])
        >>> wavelength = np.array([1.5, 1.6, 1.7])
        >>> waveguide_width, wavelength = np.meshgrid(waveguide_width, wavelength)
        >>> neff(waveguide_width, wavelength)
        array([[2. , 2.1],
               [1.9, 2. ],
               [1.8, 1.9]])

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
def neff_te0(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value  of TE0 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).
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
def neff_tm0(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value  of TM0 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).
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
def neff_te1(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value  of TE1 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).
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
def neff_tm1(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value  of TM1 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).
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
def neff_te2(width: float, wavelength: float) -> float | list[float]:
    """
    Gets Effective Index value  of TE2 by using corresponding parameters. This is
    a JAX compatible function. JIT is enabled.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar or list
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar or list

    Returns:
        Effective Index value(s).
    """
    return ndimage.map_coordinates(
        effective_index_te2,
        [
            (width - width_min) * ((width_size - 1) / (width_max - width_min)),
            (wavelength - wav_min) * ((wav_size - 1) / (wav_max - wav_min)),
        ],
        order=1,
    )
