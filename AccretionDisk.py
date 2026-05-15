#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################
import numpy as np

from physics import BlackHole, AccDisk



#######################################################################
# ------------------------ Accretion Disk Class -----------------------
#######################################################################
class SimpleDisk(AccDisk):
    def __init__(self, q):
        self.q = q
        self.BH = None
    
    def setBH(self, BH:BlackHole):
        self.BH = BH
        
    def diskEmitFreq(self, x, p):
        """
        #### Disk Emission Frequency
        : calculate the frequency of the photon emitted on the disk
        - frame of observer
        - Keplerian orbital motion assumption
        #### [Parameter]
        x : 4-position of the photon (on the disk)
        P : 4-mometum of the photon (on the disk)
        """
        t, r, theta, phi = x
        Pt, _, _, Pphi = p
        M = self.BH.M

        Ut = np.zeros_like(r)
        mask = r > 3*M
        Ut[mask] = 1/np.sqrt(1-3*M/r[mask])
        Ut[~mask] = 0
        Uphi = np.sqrt(M/(r**3))*Ut
        U_disk = np.array([Ut,
                           np.zeros_like(Ut),
                           np.zeros_like(Ut),
                           Uphi])
        # u_disk : (4, N)
        # p : (4, N)
        # ----- General Form -----
        # : not used for optimization
        # metric = self.metricTensor(x) # (4,4,N)
        # U_disk_covar = np.einsum('ij...,j...->i...', metric, U_disk) # (4, N)
        # freq_emit = -np.einsum('i...,i...->...', p, U_disk_covar) #(N)

        freq_emit = -(-(1-2*M/r)*Pt*Ut + ((r*np.sin(theta))**2)*Pphi*Uphi) # (N)
        return freq_emit
    
    def diskIntensity(self, x):
        """
        #### Disk Intensity
        : calculate the intensity emitted by the disk
        - frame of disk (need to transformated to observer frame for plotting)
        - disk model :
        $$I_emit \propto r^{-q}$$
        #### [Parameter]
        x : 4-position of the photon (on the disk)
        q : exponent of the model
        """
        t, r, theta, phi = x
        return 1/(r**self.q)