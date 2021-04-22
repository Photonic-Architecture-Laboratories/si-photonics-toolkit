import siphotonics as sip
import numpy as np
import matplotlib.pyplot as plt


def transfer_matrix_dz(width1, width2, gap, wavelength, dz):
    k12 = sip.coupling_coefficient(width1, width2, gap, wavelength)
    k21 = sip.coupling_coefficient(width2, width1, gap, wavelength)
    beta_1 = 2 * np.pi * sip.neff(width1, wavelength, 1) / wavelength
    beta_2 = 2 * np.pi * sip.neff(width2, wavelength, 1) / wavelength
    beta_0 = np.sqrt((beta_1 / 2 - beta_2 / 2) ** 2 + k12 * k21)

    _transfer_matrix = np.zeros((2, 2), dtype=complex)
    _transfer_matrix[0][0] = np.cos(beta_0 * dz) - 1j * (-1 * (beta_2 - beta_1) / (2 * beta_0)) * np.sin(beta_0 * dz)
    _transfer_matrix[0][1] = +1j * k12 / beta_0 * np.sin(beta_0 * dz)
    _transfer_matrix[1][0] = +1j * k21 / beta_0 * np.sin(beta_0 * dz)
    _transfer_matrix[1][1] = np.cos(beta_0 * dz) + 1j * (-1 * (beta_2 - beta_1) / (2 * beta_0)) * np.sin(beta_0 * dz)
    return _transfer_matrix  # * np.exp(-1j*((beta_1-beta_2)/2)*dz)   # not sure about this


def transfer_matrix(width1, width2, gap, wavelength, L, dz):
    if not isinstance(width1, (list, np.ndarray)):
        raise TypeError("width1 has wrong data type.")
    if not isinstance(width2, (list, np.ndarray)):
        raise TypeError("width2 has wrong data type.")
    if not isinstance(gap, (list, np.ndarray)):
        raise TypeError("gap has wrong data type.")
    if not isinstance(wavelength, (list, np.ndarray)):
        raise TypeError("wavelength has wrong data type.")

    if not (len(width1) == int(L / dz)):
        raise ValueError("width1 has incompatible length " + str(len(width1)) + " instead of " + str(int(L / dz)))
    if not (len(width2) == int(L / dz)):
        raise ValueError("width2 has incompatible length " + str(len(width2)) + " instead of " + str(int(L / dz)))
    if not (len(gap) == int(L / dz)):
        raise ValueError("gap has incompatible length " + str(len(gap)) + " instead of " + str(int(L / dz)))

    sample_point_number = int(L / dz)
    tm_current = np.zeros((len(wavelength), sample_point_number, 2, 2), dtype=complex)

    for i, wl in enumerate(wavelength):
        tm_current[i, 0, :, :] = transfer_matrix_dz(width1[0], width2[0], gap[0], wl, dz)
        for j in range(1, sample_point_number):
            tm_i = transfer_matrix_dz(width1[j], width2[j], gap[j], wl, dz)
            tm_current[i, j, :, :] = np.dot(tm_i, tm_current[i, j - 1, :, :])
    return tm_current


def plot_power_output(transfer_mat, wavelength, device_length, kind="frequency_sweep"):
    plt.rcParams['figure.figsize'] = [11, 6]
    plt.rcParams.update({'font.size': 11})

    wavelength_size = wavelength.shape[0]
    input = np.array([[1], [0]], dtype=complex)

    if kind == "device_length":
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0.02})
        fig.suptitle("Power Ratio of Through and Cross Port")

        sample_point_number = transfer_mat.shape[1]
        device_length_array = np.linspace(0, device_length, sample_point_number)
        t = np.zeros((wavelength_size, sample_point_number), dtype=complex)
        q = np.zeros((wavelength_size, sample_point_number), dtype=complex)

        for i in range(wavelength.shape[0]):
            for j in range(sample_point_number):
                t[i, j] = np.dot(transfer_mat[i, j, :, :], input)[0][0]
                q[i, j] = np.dot(transfer_mat[i, j, :, :], input)[1][0]
            axs[0].plot(device_length_array, t[i].real ** 2 + t[i].imag ** 2,
                        label="$t^{2}$ - " + "{0:.04}".format(wavelength[i]) + " $\mu{m}$ wavelength")
            axs[0].legend()
            axs[1].plot(device_length_array, q[i].real ** 2 + q[i].imag ** 2,
                        label="$q^{2}$ - " + "{0:.04}".format(wavelength[i]) + " $\mu{m}$ wavelength")
            axs[1].legend()
        axs[1].set_xlabel('Device Length ($\mu{m}$)')
        axs[0].set_ylabel("$t^{2}$")
        axs[1].set_ylabel("$q^{2}$")
        for ax in axs:
            ax.label_outer()
            ax.grid()

    elif kind == "frequency_sweep":
        t = np.zeros((wavelength_size, 1), dtype=complex)
        q = np.zeros((wavelength_size, 1), dtype=complex)

        for i in range(wavelength.shape[0]):
            t[i, 0] = np.dot(transfer_mat[i, -1, :, :], input)[0][0]
            q[i, 0] = np.dot(transfer_mat[i, -1, :, :], input)[1][0]
        plt.plot(wavelength, t.real ** 2 + t.imag ** 2, label="$t^{2}$")
        plt.plot(wavelength, q.real ** 2 + q.imag ** 2, label="$q^{2}$")
        plt.title("Power Ratio of Through and Cross Port at Device Length of " + str(device_length) + " $\mu{m}$")
        plt.xlabel("Wavelength ($\mu{m}$)")

        plt.ylabel("Power Ratio")
        plt.legend()
        plt.grid()
    plt.show()
