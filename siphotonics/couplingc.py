import os
from scipy.interpolate import interp1d
import numpy as np
import pickle

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))
with open(r"coupling_coefficient_TE0-TE0.pickle", "rb") as input_file:
    couplingCoefficientArray = pickle.load(input_file)
os.chdir(user_dir)

gapArr = np.logspace(np.log(100), np.log(2001), num=150, endpoint=True, base=np.e, dtype=int)

def couplingCoefficient(width1, width2, gap, wavelength):
    iSource = interp1d(np.arange(300,701,10), couplingCoefficientArray, axis=0)
    iTarget = interp1d(np.arange(300,701,10), iSource(width2*1000), axis=0)
    iGap    = interp1d(gapArr, iTarget(width1*1000), axis=0)
    iWl     = interp1d(np.arange(1200, 1701, 10), iGap(gap*1000), axis=0)
    return iWl(wavelength*1000)
