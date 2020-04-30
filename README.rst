siphotonics
===========

Guide for Local Usage
=====================

Windows Users
-------------

* Make sure you download Python with adding it to PATH at download step.
* Control if "pip" is installed or not by typing ``pip -V``.

    * If "pip" is not installed, type ``python get-pip.py``.
    
* Go to the directory where you clone this repo.
* Type ``pip install .``
* Done.

Linux Users
-----------
* Go to the directory where you clone this repo.
* Type ``pip install .``
* Done.

Guide of Functions:
======

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
