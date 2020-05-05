siphotonics
-----------

USAGE:

### Effective Index Values of First Five Modes
----------------------------------------------
    >>> siphotonics.neff(500,  1550, "te1")
    >>> siphotonics.neff(500,  1550, "tM0")
    >>> siphotonics.neff(500,  1550, "TE1")
    >>> siphotonics.neff(500,  1550,     1)
    >>> siphotonics.neff(500,  1550,     2)
                          ^      ^      ^
                       width  wavelng  mode
==========================================================
### Polarization Fraction of First Five Modes
---------------------------------------------
    >>> siphotonics.polarization_frac(500,  1550, "tm1")
    >>> siphotonics.polarization_frac(500,  1550, "tE0")
    >>> siphotonics.polarization_frac(500,  1550, "Te1")
    >>> siphotonics.polarization_frac(500,  1550,     1)
    >>> siphotonics.polarization_frac(500,  1550,     5)
                                       ^      ^      ^
                                    width  wavelng  mode
==========================================================
### Permittivity of Si & SiO2
-----------------------------
    >>> siphotonics.perm_si(1550)
    >>> siphotonics.perm_oxide(1550)
                                ^
                            wavelength
==========================================================