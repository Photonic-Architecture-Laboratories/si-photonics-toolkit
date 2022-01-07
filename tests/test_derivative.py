import numpy as np
import pytest

import siphotonics as sip


def test_derivative():
    """
    Test derivative function
    :return:
    """
    x_interval = np.linspace(0, 10, 400)
    cos_func = np.cos(2 * np.pi * x_interval)

    dydx = sip.derivative(cos_func, order=1)
    dydx2 = sip.derivative(cos_func, order=2)
    dydx3 = sip.derivative(cos_func, order=3)
    dydx4 = sip.derivative(cos_func, order=4)

    assert cos_func.shape == dydx.shape
    assert cos_func.shape == dydx2.shape
    assert cos_func.shape == dydx3.shape
    assert cos_func.shape == dydx4.shape

    with pytest.raises(Exception):
        assert cos_func.shape == sip.derivative(cos_func, order=5)
