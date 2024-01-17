import numpy as np


def magnetization2d(T):
    if T < 1:
        return 1
    return max(0, np.power(1-np.power( (1-T**2)/(2*T) ,4), 1/8))


def magnetization1d(T, h_J):
    return np.sinh(h_J/T) / np.sqrt( np.sinh(h_J/T)**2 + np.exp(-4*(1/T)) ) 


def cv2d(T, Tc, B):
    return B*np.log(abs(T-Tc))




