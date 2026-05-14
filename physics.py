#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod

from numerical import RK4, RK4Batch



#######################################################################
# --------------------------- Abstract Class --------------------------
#######################################################################
class BlackHole(ABC):
    def __init__(self, M, Q, J):
        super().__init__()
        self.M = M
        self.Q = Q
        self.J = J

    @abstractmethod
    def metricTensor(self, x_sph): pass

    @abstractmethod
    def vierbein(self, x_sph): pass
    
    @abstractmethod
    def geodesic(self, l, y): pass

class AccDisk(ABC):
    def __init__(self): pass

    @abstractmethod
    def setBH(self, BH): pass



#######################################################################
# ----------------------- Physical Law Functions ----------------------
#######################################################################

def photonFreq(mom, freq):
    """
    #### Photon Momentum Scaler
    : given 4-momentum of photon, scale the momentum for photon to have given frequency
    #### [Parameter]
    mom : 4-momentum
    freq : frequency
    * natural unit : $c=\hbar=G=1$
    """
    scale = freq/mom[0]
    return scale*mom