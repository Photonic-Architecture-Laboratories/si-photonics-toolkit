import pickle
import time
import os

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))

with open(r"neff_pol.pickle", "rb") as input_file:
    f = pickle.load(input_file)

F = f[0]
P = f[1]

neff_te_list =[]
neff_tm_list = []
pol_te_list = []
pol_tm_list = []

for i in range(5):
    pol_te_list.append({})
    pol_tm_list.append({})
    neff_te_list.append({})
    neff_tm_list.append({})

os.chdir(user_dir)

def neff(width, wavelength, te_or_tm):
    if not (width >= 250 and width <= 700):
            raise ValueError("Width must be between 250-700")
    if not (wavelength >=1200 and wavelength <= 1700):
            raise ValueError("Wavelength must be between 1200-1700")
            
    if isinstance(te_or_tm, str):

        if te_or_tm[0:2] == "te":
            if wavelength in neff_te_list[int(te_or_tm[2:3])]:
                if width in neff_te_list[int(te_or_tm[2:3])][wavelength]:
                    return neff_te_list[int(te_or_tm[2:3])][wavelength][width]
        if te_or_tm[0:2] == "tm":
            if wavelength in neff_tm_list[int(te_or_tm[2:3])] and width in neff_tm_list[int(te_or_tm[2:3])][wavelength]:
                    return neff_tm_list[int(te_or_tm[2:3])][wavelength][width]

        te_or_tm = te_or_tm.lower()
        last_index = 5
        te_number = 0
        tm_number = 0
        
        for i in range(len(F)):
            if F[i](wavelength, width)[0] < 1.44:
                last_index = i
                break
            if P[i](wavelength, width)[0] > 0.5:
                te_number += 1
            if P[i](wavelength, width)[0] < 0.5:
                tm_number += 1
        
        neff_te = []
        neff_tm = []
        for i in range(last_index):
            if P[i](wavelength, width)[0] > 0.50:
                neff_te.append(F[i](wavelength, width)[0])
            else:
                neff_tm.append(F[i](wavelength, width)[0])

        if te_or_tm[0:2] == "te" and int(te_or_tm[2:3]) < te_number:
            ind = int(te_or_tm[2:3])
            a = {wavelength: {width : neff_te[ind]}}
            neff_te_list[ind].update(a)
            return neff_te[ind]
        else:
            if not te_or_tm[0:2] == "tm":
                raise ValueError("There is no TE" + te_or_tm[2:3] + " for Wavelength: " + str(wavelength) + " & Width: " + str(width))

        if te_or_tm[0:2] == "tm" and int(te_or_tm[2:3]) < tm_number:
            ind = int(te_or_tm[2:3])
            b = {wavelength: {width : neff_tm[ind]}}
            neff_tm_list[ind].update(b)
            return neff_tm[ind]
        else:
            if not te_or_tm[0:2] == "te":
                raise ValueError("There is no TM" + te_or_tm[2:3] + " for Wavelength: " + str(wavelength) + " & Width: " + str(width))
    
    ## FOR MODE NUMBER AS INT (1, 2, 3, 4, 5)
    if isinstance(te_or_tm, int):
        if not (te_or_tm >=1 and te_or_tm<=5):
            raise ValueError("Mode should be between 1-5")
        if te_or_tm == 1:
            return F[0](wavelength, width)[0]
        elif te_or_tm == 2:
            return F[1](wavelength, width)[0]
        elif te_or_tm == 3:
            return F[2](wavelength, width)[0]
        elif te_or_tm == 4:
            return F[3](wavelength, width)[0]
        else:
            return F[4](wavelength, width)[0]

def polarization_frac(width, wavelength, te_or_tm):

    if not (width >= 250 and width <= 700):
            raise ValueError("Width must be between 250-700")
    if not (wavelength >=1200 and wavelength <= 1700):
            raise ValueError("Wavelength must be between 1200-1700")

    if isinstance(te_or_tm, str):

        if te_or_tm[0:2] == "te":
            if wavelength in pol_te_list[int(te_or_tm[2:3])]:
                if width in pol_te_list[int(te_or_tm[2:3])][wavelength]:
                    return pol_te_list[int(te_or_tm[2:3])][wavelength][width]
        if te_or_tm[0:2] == "tm":
            if wavelength in pol_tm_list[int(te_or_tm[2:3])] and width in pol_tm_list[int(te_or_tm[2:3])][wavelength]:
                    return pol_tm_list[int(te_or_tm[2:3])][wavelength][width]

        te_or_tm = te_or_tm.lower()
        last_index = 5
        te_number = 0
        tm_number = 0
        
        for i in range(len(F)):
            if F[i](wavelength, width)[0] < 1.44:
                last_index = i
                break
            if P[i](wavelength, width)[0] > 0.5:
                te_number += 1
            if P[i](wavelength, width)[0] < 0.5:
                tm_number += 1
        
        pol_te = []
        pol_tm = []
        for i in range(last_index):
            if P[i](wavelength, width)[0] > 0.50:
                pol_te.append(P[i](wavelength, width)[0])
            else:
                pol_tm.append(P[i](wavelength, width)[0])

        if te_or_tm[0:2] == "te" and int(te_or_tm[2:3]) < te_number:
            ind = int(te_or_tm[2:3])
            a = {wavelength: {width : pol_te[ind]}}
            pol_te_list[ind].update(a)
            return pol_te[ind]
        else:
            if not te_or_tm[0:2] == "tm":
                raise ValueError("There is no TE" + te_or_tm[2:3] + " for Wavelength: " + str(wavelength) + " & Width: " + str(width))

        if te_or_tm[0:2] == "tm" and int(te_or_tm[2:3]) < tm_number:
            ind = int(te_or_tm[2:3])
            b = {wavelength: {width : pol_tm[ind]}}
            pol_tm_list[ind].update(b)
            return pol_tm[ind]
        else:
            if not te_or_tm[0:2] == "te":
                raise ValueError("There is no TM" + te_or_tm[2:3] + " for Wavelength: " + str(wavelength) + " & Width: " + str(width))

    if isinstance(te_or_tm, int):
        if not (te_or_tm >=1 and te_or_tm<=5):
            raise ValueError("Mode should be between 1-5")
        if te_or_tm == 1:
            return P[0](wavelength, width)[0]
        elif te_or_tm == 2:
            return P[1](wavelength, width)[0]
        elif te_or_tm == 3:
            return P[2](wavelength, width)[0]
        elif te_or_tm == 4:
            return P[3](wavelength, width)[0]
        else:
            return P[4](wavelength, width)[0]
