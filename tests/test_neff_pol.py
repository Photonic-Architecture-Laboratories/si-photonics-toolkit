import siphotonics as sip


def test_polarization_fraction():
    """
    Test polarization fraction function
    :return:
    """
    assert sip.polarization_frac(0.5, 1.55, "te0") > 0.5
    assert sip.polarization_frac(0.5, 1.55, "tm0") < 0.5
