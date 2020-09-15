import numpy as np

def derivative(f, order=1, step=1):
    if order == 1:
        first_der = np.zeros(len(f))
        first_der[0] = (f[1] - f[0]) / 1
        first_der[1] = (-1*f[0] + 1*f[2]) / 2
        first_der[-1] = (f[-1] - f[-2]) / 1
        first_der[-2] = (-1*f[-3] + 1*f[-1]) / 2
        for i in range(2,len(f)-2,1):
            first_der[i] = (1*f[i - 2] - 8*f[i - 1] + 8*f[i + 1] - 1*f[i + 2]) / 12
        return first_der/step
    elif order == 2:
        second_der = np.zeros(len(f))
        second_der[0] = (1*f[0] - 2*f[1] + 1*f[2]) / 1
        second_der[1] = (2*f[0] - 5*f[1] + 4*f[2] - 1*f[3]) / 1
        second_der[-1] = (1*f[-3] - 2*f[-2] + 1*f[-1]) / 1
        second_der[-2] = (0*f[-4] + 1*f[-3] - 2*f[-2] + 1*f[-1]) / 1
        for i in range(2,len(f)-2,1):
            second_der[i] = (-1*f[i - 2] + 16*f[i - 1] - 30*f[i] + 16*f[i + 1] - 1*f[i + 2]) / 12
        return second_der/(step**2)
    elif order == 3:
        third_der = np.zeros(len(f))
        third_der[0] = (-1*f[0] + 3*f[1] - 3*f[2] + 1*f[3]) / 1
        third_der[1] = (-1*f[0] + 3*f[1] - 3*f[2] + 1*f[3]) / 1
        third_der[-1] = (-1*f[-4] + 3*f[-3] - 3*f[-2] + 1*f[-1]) / 1
        third_der[-2] = (5*f[-5] - 22*f[-4] + 36*f[-3] - 26*f[-2] + 7*f[-1]) / 2
        for i in range(2,len(f)-2,1):
            third_der[i] = (-1*f[i - 2] + 2*f[i - 1] + 0*f[i] - 2*f[i + 1] + 1*f[i + 2]) / 2
        return third_der/(step**3)
    elif order == 4:
        fourth_der = np.zeros(len(f))
        fourth_der[0] = (1*f[0] - 4*f[1] + 6*f[2] - 4*f[3] + 1*f[4]) / 1
        fourth_der[1] = (2*f[0] - 9*f[1] + 16*f[2] - 14*f[3] + 6*f[4] - 1*f[5]) / 1
        fourth_der[-1] = (1*f[-5] - 4*f[-4] + 6*f[-3] - 4*f[-2] + 1*f[-1]) / 1
        fourth_der[-2] = (-1*f[-6] + 6*f[-5] - 14*f[-4] + 16*f[-3] - 9*f[-2] + 2*f[-1]) / 1
        for i in range(2,len(f)-2,1):
            fourth_der[i] = (1*f[i - 2] - 4*f[i - 1] + 6*f[i] - 4*f[i + 1] + 1*f[i + 2]) / 1
        return fourth_der/(step**4)