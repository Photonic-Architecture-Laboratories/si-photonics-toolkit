import pickle
import os

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open('permittivity.pickle', 'rb') as handle:
    perm = pickle.load(handle)
os.chdir(user_dir)


def perm_si(wavelength):
    if not (wavelength >=1200 and wavelength <= 1700):
        raise ValueError("Wavelength must be between 1200-1700")
    
    return perm["Si"](wavelength)

def perm_oxide(wavelength):
    if not (wavelength >=1200 and wavelength <= 1700):
        raise ValueError("Wavelength must be between 1200-1700")
    
    return perm["SiO2"](wavelength)
    