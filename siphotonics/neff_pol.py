import os
import pickle

from siphotonics.effective_index import width_min, width_max, wav_min, wav_max

user_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), "data"))

with open(r"neff_pol.pickle", "rb") as input_file:
    f = pickle.load(input_file)

F = f[0]
P = f[1]

neff_te_list = []
neff_tm_list = []
pol_te_list = []
pol_tm_list = []

for i in range(5):
    pol_te_list.append({})
    pol_tm_list.append({})
    neff_te_list.append({})
    neff_tm_list.append({})

os.chdir(user_dir)


def polarization_frac(width, wavelength, te_or_tm):
    """
    Polarization fraction of `TE` or `TM` mode of light at a specific wavelength in a waveguide with a specific width.
    :param width:
    :param wavelength:
    :param te_or_tm:
    :return:
    """
    width_nm = width * 1000
    wavelength_nm = wavelength * 1000
    if not width_min <= width <= width_max:
        raise ValueError(f"Width must be between {width_min}-{width_max} micron")
    if not wav_min <= wavelength <= wav_max:
        raise ValueError(f"wavelength must be between {wav_min}-{wav_max} micron")

    if isinstance(te_or_tm, str):

        if te_or_tm[0:2] == "te":
            if wavelength_nm in pol_te_list[int(te_or_tm[2:3])]:
                if width_nm in pol_te_list[int(te_or_tm[2:3])][wavelength_nm]:
                    return pol_te_list[int(te_or_tm[2:3])][wavelength_nm][width_nm]
        if te_or_tm[0:2] == "tm":
            if (
                wavelength_nm in pol_tm_list[int(te_or_tm[2:3])]
                and width_nm in pol_tm_list[int(te_or_tm[2:3])][wavelength_nm]
            ):
                return pol_tm_list[int(te_or_tm[2:3])][wavelength_nm][width_nm]

        te_or_tm = te_or_tm.lower()
        last_index = 5
        te_number = 0
        tm_number = 0

        for j, _f in enumerate(F):
            if _f(wavelength_nm, width_nm)[0] < 1.44:
                last_index = j
                break
            if P[j](wavelength_nm, width_nm)[0] > 0.5:
                te_number += 1
            if P[j](wavelength_nm, width_nm)[0] < 0.5:
                tm_number += 1

        pol_te = []
        pol_tm = []
        for j in range(last_index):
            if P[j](wavelength_nm, width_nm)[0] > 0.50:
                pol_te.append(P[j](wavelength_nm, width_nm)[0])
            else:
                pol_tm.append(P[j](wavelength_nm, width_nm)[0])

        if te_or_tm[0:2] == "te" and int(te_or_tm[2:3]) < te_number:
            ind = int(te_or_tm[2:3])
            a_dict = {wavelength_nm: {width_nm: pol_te[ind]}}
            pol_te_list[ind].update(a_dict)
            return pol_te[ind]
        if not te_or_tm[0:2] == "tm":
            raise ValueError(
                "There is no TE" + te_or_tm[2:3] + " for wavelength: " + str(wavelength_nm) + " & Width: " + str(width_nm)
            )

        if te_or_tm[0:2] == "tm" and int(te_or_tm[2:3]) < tm_number:
            ind = int(te_or_tm[2:3])
            a_dict = {wavelength_nm: {width_nm: pol_tm[ind]}}
            pol_tm_list[ind].update(a_dict)
            return pol_tm[ind]

        if not te_or_tm[0:2] == "te":
            raise ValueError(
                "There is no TM" + te_or_tm[2:3] + " for wavelength: " + str(wavelength_nm) + " & Width: " + str(width_nm)
            )

    if isinstance(te_or_tm, int):
        if not 1 <= te_or_tm <= 5:
            raise ValueError("Mode should be between 1-5")
        if te_or_tm == 1:
            return P[0](wavelength_nm, width_nm)[0]
        if te_or_tm == 2:
            return P[1](wavelength_nm, width_nm)[0]
        if te_or_tm == 3:
            return P[2](wavelength_nm, width_nm)[0]
        if te_or_tm == 4:
            return P[3](wavelength_nm, width_nm)[0]
        return P[4](wavelength_nm, width_nm)[0]
    return -1
