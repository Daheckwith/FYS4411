import numpy as np
from numba import njit

@njit
def Qfac(r,b,w, NumberHidden, sigma= 1):
    Q = np.zeros((NumberHidden), np.double)
    temp = np.zeros((NumberHidden), np.double)
    
    for ih in range(NumberHidden):
        temp[ih] = (r*w[:,:,ih]).sum()
        
    Q = b + temp/sigma**2
    
    return Q
    