import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pickle

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open(r"coupling_coefficient_TE0-TE0.pickle", "rb") as input_file:
    couplingCoefficientArray = pickle.load(input_file)
os.chdir(user_dir)

width1_array = np.linspace(300, 701, 41)
width2_array = np.linspace(300, 701, 41)
wavelengthArray = np.arange(1200, 1701, 10)
gapArr = np.logspace(np.log(100), np.log(2001), num=150, endpoint=True, base=np.e, dtype=int)
points = (width1_array, width2_array, gapArr, wavelengthArray)
my_interp = RegularGridInterpolator(points, couplingCoefficientArray)

def coupling_coefficient(width1, width2, gap, wavelength):
    return my_interp([width1*1000,width2*1000,gap*1000,wavelength*1000])
