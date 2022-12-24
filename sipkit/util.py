from __future__ import annotations
import os

from jax import config
from jax import numpy as jnp

MODES = ["te0", "te1", "te2", "tm0", "tm1"]

config.update("jax_enable_x64", True)


def _read_effective_index(file_name: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reads effective index data from file.

    Args:
        file_name (str): Name of the file.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Effective index data, width data, wavelength data.
    """
    
    with open(os.path.join(os.path.dirname(__file__), "data", file_name), "r") as _file:
        _lines = _file.readlines()

    _neff_data = jnp.array(list(map(float, _lines[1][:-2].split(","))))
    _width_data = jnp.array(list(map(float, _lines[3][:-2].split(","))))
    _wav_data = jnp.array(list(map(float, _lines[5][:-2].split(","))))

    _wav_size = _wav_data.shape[0]
    _width_size = _width_data.shape[0]

    if any(mode in file_name for mode in MODES):
        _neff_data = _neff_data.reshape((_width_size, _wav_size))
    else:
        _neff_data = _neff_data.reshape((_wav_size, _width_size))
    return _neff_data, _width_data, _wav_data
