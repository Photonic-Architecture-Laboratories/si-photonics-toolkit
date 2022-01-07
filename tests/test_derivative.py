import numpy as np
import pytest

import siphotonics as sip


def test_derivative():
    x = np.linspace(0, 10, 400)
    y = np.cos(2 * np.pi * x)

    dydx = sip.derivative(y, order=1)
    dydx2 = sip.derivative(y, order=2)
    dydx3 = sip.derivative(y, order=3)
    dydx4 = sip.derivative(y, order=4)

    assert y.shape == dydx.shape
    assert y.shape == dydx2.shape
    assert y.shape == dydx3.shape
    assert y.shape == dydx4.shape

    with pytest.raises(Exception):
        assert y.shape == sip.derivative(y, order=5)
