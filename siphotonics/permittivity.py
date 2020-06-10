import pickle
import os

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open('permittivity.pickle', 'rb') as handle:
    perm = pickle.load(handle)
os.chdir(user_dir)


def perm_si(wavelength):
    if not (wavelength >=1.2 and wavelength <= 1.7):
        raise ValueError("Wavelength must be between 1.2-1.7 micron")
    
    return perm["Si"](wavelength * 1000)

def perm_oxide(wavelength):
    if not (wavelength >=1.2 and wavelength <= 1.2):
        raise ValueError("Wavelength must be between 1.2-1.7 micron")
    
    return perm["SiO2"](wavelength * 1000)
    