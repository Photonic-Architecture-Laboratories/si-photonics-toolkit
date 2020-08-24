import siphotonics as sip 
import numpy as np

def fsr(wavelength, radius, width):
    return (wavelength**2) / (2 * np.pi * radius * sip.ng(width, wavelength))
