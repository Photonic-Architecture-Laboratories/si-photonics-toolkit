from siphotonics.trax_neff import neff


def test_neff():
    assert neff(0.5, 1.55) < 3.5
