import numpy as np


def derivative(func, order=1, step=1):
    """
    Derivative of given array-like data structure up to 4th order.
    :param func:
    :param order:
    :param step:
    :return:
    """
    if order == 1:
        first_der = np.zeros(len(func))
        first_der[0] = (func[1] - func[0]) / 1
        first_der[1] = (-1 * func[0] + 1 * func[2]) / 2
        first_der[-1] = (func[-1] - func[-2]) / 1
        first_der[-2] = (-1 * func[-3] + 1 * func[-1]) / 2
        for i in range(2, len(func) - 2, 1):
            first_der[i] = (1 * func[i - 2] - 8 * func[i - 1] + 8 * func[i + 1] - 1 * func[i + 2]) / 12
        return first_der / step
    if order == 2:
        second_der = np.zeros(len(func))
        second_der[0] = (1 * func[0] - 2 * func[1] + 1 * func[2]) / 1
        second_der[1] = (2 * func[0] - 5 * func[1] + 4 * func[2] - 1 * func[3]) / 1
        second_der[-1] = (1 * func[-3] - 2 * func[-2] + 1 * func[-1]) / 1
        second_der[-2] = (0 * func[-4] + 1 * func[-3] - 2 * func[-2] + 1 * func[-1]) / 1
        for i in range(2, len(func) - 2, 1):
            second_der[i] = (
                -1 * func[i - 2] + 16 * func[i - 1] - 30 * func[i] + 16 * func[i + 1] - 1 * func[i + 2]
            ) / 12
        return second_der / (step ** 2)
    if order == 3:
        third_der = np.zeros(len(func))
        third_der[0] = (-1 * func[0] + 3 * func[1] - 3 * func[2] + 1 * func[3]) / 1
        third_der[1] = (-1 * func[0] + 3 * func[1] - 3 * func[2] + 1 * func[3]) / 1
        third_der[-1] = (-1 * func[-4] + 3 * func[-3] - 3 * func[-2] + 1 * func[-1]) / 1
        third_der[-2] = (5 * func[-5] - 22 * func[-4] + 36 * func[-3] - 26 * func[-2] + 7 * func[-1]) / 2
        for i in range(2, len(func) - 2, 1):
            third_der[i] = (-1 * func[i - 2] + 2 * func[i - 1] + 0 * func[i] - 2 * func[i + 1] + 1 * func[i + 2]) / 2
        return third_der / (step ** 3)
    if order == 4:
        fourth_der = np.zeros(len(func))
        fourth_der[0] = (1 * func[0] - 4 * func[1] + 6 * func[2] - 4 * func[3] + 1 * func[4]) / 1
        fourth_der[1] = (2 * func[0] - 9 * func[1] + 16 * func[2] - 14 * func[3] + 6 * func[4] - 1 * func[5]) / 1
        fourth_der[-1] = (1 * func[-5] - 4 * func[-4] + 6 * func[-3] - 4 * func[-2] + 1 * func[-1]) / 1
        fourth_der[-2] = (
            -1 * func[-6] + 6 * func[-5] - 14 * func[-4] + 16 * func[-3] - 9 * func[-2] + 2 * func[-1]
        ) / 1
        for i in range(2, len(func) - 2, 1):
            fourth_der[i] = (1 * func[i - 2] - 4 * func[i - 1] + 6 * func[i] - 4 * func[i + 1] + 1 * func[i + 2]) / 1
        return fourth_der / (step ** 4)
    return -1
