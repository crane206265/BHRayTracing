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
    def __init__(self, M, J, Q):
        super().__init__()
        self.M = M
        self.J = J
        self.Q = Q

    @abstractmethod
    def metricTensor(self, x): pass

    @abstractmethod
    def vierbein_l2g(self, x): pass

    @abstractmethod
    def vierbein_g2l(self, x): pass
    
    @abstractmethod
    def geodesic(self, l, y): pass

class AccDisk(ABC):
    def __init__(self): pass

    @abstractmethod
    def setBH(self, BH): pass

    @abstractmethod
    def diskEmitFreq(self, x, p): pass

    @abstractmethod
    def diskIntensity(self, x): pass



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