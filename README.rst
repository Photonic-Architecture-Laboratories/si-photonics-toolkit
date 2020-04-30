#siphotonics
-----------

##USAGE:

    >>> siphotonics.neff(500,  1550, "te1")
    >>> siphotonics.neff(500,  1550, "tM0")
    >>> siphotonics.neff(500,  1550, "TE1")
    >>> siphotonics.neff(500,  1550,     1)
    >>> siphotonics.neff(500,  1550,     2)
                          ^      ^      ^
                       width  wavelng  mode
==========================================================

    >>> siphotonics.polarization_frac(500,  1550, "tm1")
    >>> siphotonics.polarization_frac(500,  1550, "tE0")
    >>> siphotonics.polarization_frac(500,  1550, "Te1")
    >>> siphotonics.polarization_frac(500,  1550,     1)
    >>> siphotonics.polarization_frac(500,  1550,     5)
                                       ^      ^      ^
                                    width  wavelng  mode
==========================================================
