# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from physics import Schwarzchild
from numerical import RK4
from utils import plot3DSchwarzchild


# ------------------------------ Param.s ------------------------------

M = 1               # Mass of BH
r_screen = 30       # BH - obs distance (unit : M)
FOV = 2            # Field of View (FOV) (unit : deg)
PPD = 10            # Pixel per Deg. (PPD)


# --------------------------- Param. Setting --------------------------
r_screen *= M
img = np.zeros((FOV*PPD, FOV*PPD))



# ------------------------------ Main ------------------------------

ax = None
BH = Schwarzchild(M=M)

x_screen = np.array([r_screen, np.pi/4, 0])
freq = 1.3
BH.screenInitSetting(x_screen, freq=freq, FOV=FOV, PPD=PPD)

N = 15
init_mom_arr = BH.screenSampling(N)
for i in tqdm(range(N)):
    init_pos = np.array([0, x_screen[0], x_screen[1], x_screen[2]])
    init_mom = init_mom_arr[:, i]
    y_init = np.concatenate((init_pos, init_mom), axis=0)

    geodesic = lambda l, y: BH.geodesic(l, y)
    terminalCondition = lambda y, h: BH.terminalCondition(y, h)
    _, Y = RK4(geodesic, t_range=(0, 500), initial=y_init, h=1e-1, terminalCondition=terminalCondition)
    x_sph = Y[:, 1:4]

    freq = init_mom[0]
    ax = plot3DSchwarzchild(x_sph, ax=ax, M=M, freq=freq, lim_set=(-r_screen, r_screen))
plt.show()