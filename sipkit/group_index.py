from __future__ import annotations

import jaxlib
from jax import grad

from sipkit.effective_index import neff


def ng(width: float, wavelength: float) -> jaxlib.xla_extension.DeviceArray | jaxlib.xla_extension.Array:
    """
    Group Index of light at a specified wavelength in a waveguide with a specified width.

    Args:
        width (float): Waveguide width in microns. (0.24 - 0.7) Scalar
        wavelength (float): Wavelength in microns. (1.2 - 1.7) Scalar

    Returns:
        Group Index value.

    Examples:
        >>> group_index(0.5, 1.5)
        Array(0.5, dtype=float32)


    """
    return neff(width, wavelength) - wavelength * grad(neff, argnums=1)(width, wavelength)
