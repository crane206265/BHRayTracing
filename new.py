# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt

import physics as PH
from numerical import RK4
from utils import plot3D


# ------------------------------ Param.s ------------------------------

M = 1               # Mass of BH
r_screen = 50       # BH - obs distance (unit : M)
FOV = 20            # Field of View (FOV) (unit : deg)
PPD = 10            # Pixel per Deg. (PPD)


# --------------------------- Param. Setting --------------------------
r_screen *= M
img = np.zeros((FOV*PPD, FOV*PPD))



# ------------------------------ Main ------------------------------

ax = None
for i in range(10):
    init_pos = np.array([0, 10, np.pi/2, 0])
    init_mom = np.array([None, -1, 0.1*np.random.random(1)[0]-0.05, 0.1*np.random.random(1)[0]-0.05])
    init_mom[0] = PH.photonMomentumSchwarzchild(init_pos[1:], init_mom[1:])
    y_init = np.array([0, 10, np.pi/2, 0, 1, -1, 0.1*np.random.random(1)[0]-0.05, 0.1*np.random.random(1)[0]-0.05])
    geodesic = lambda l, y: PH.geodesicSchwarzchild(l, y, M=M)
    terminalCondition = lambda y, h: PH.terminalConditionSchwarzchild(y, h, M=M)
    _, Y = RK4(geodesic, t_range=(0, 500), initial=y_init, h=1e-2, terminalCondition=terminalCondition)
    x_sph = Y[:, 1:4]

    ax = plot3D(x_sph, ax=ax, M=M)
plt.show()
