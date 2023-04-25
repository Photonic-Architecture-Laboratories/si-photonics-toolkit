import jax.numpy as jnp

import sipkit as sip


def test_permittivity():
    """ Test permittivity.py
    """
    assert sip.perm_si(1.5) > 0
    assert sip.perm_oxide(1.5) > 0
    assert sip.perm_si(1.5) > sip.perm_oxide(1.5)

    assert len(sip.perm_si([1.5, 1.6])) == 2
    assert len(sip.perm_oxide([1.5, 1.6])) == 2
    assert sip.perm_si([1.5, 1.6])[0] > 0

    assert len(sip.perm_si(jnp.array([1.5, 1.6]))) == 2
    assert len(sip.perm_oxide(jnp.array([1.5, 1.6]))) == 2
