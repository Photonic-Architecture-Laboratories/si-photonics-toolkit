from siphotonics.trax_neff import neff


def test_neff():
    """
    Test effective index function.
    :return:
    """
    assert neff(0.5, 1.55) < 3.5
