#siphotonics
-----------

##USAGE:

    >>> siphotonics.neff(500, 1550, "te1")
    >>> siphotonics.neff(500, 1550, "tE0")
    >>> siphotonics.neff(500, 1550, "TE1")
    >>> siphotonics.neff(500, 1550,     0)
    >>> siphotonics.neff(500, 1550,     1)
                          ^     ^     ^
                       width wavelng  mode
==========================================================

    >>> siphotonics.polarization_frac(500, 1550, "te1")
    >>> siphotonics.polarization_frac(500, 1550, "tE0")
    >>> siphotonics.polarization_frac(500, 1550, "TE1")
    >>> siphotonics.polarization_frac(500, 1550,     0)
    >>> siphotonics.polarization_frac(500, 1550,     1)
                                       ^     ^     ^
                                    width wavelng  mode
==========================================================

