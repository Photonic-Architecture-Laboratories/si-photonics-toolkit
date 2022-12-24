import sipkit as sip


def test_neff():
    """
    Test effective index function.
    :return:
    """
    assert sip.effective_index.neff(0.5, 1.55) < 3.5
