# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from physics import Schwarzchild
from numerical import RK4
from utils import plot2DSchwarzchild, plot3DSchwarzchild


# --------------------------- Param. Setting --------------------------
SAVE_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\BHRayTracing\saved_data"


M = 1               # Mass of BH
r_screen = 200      # BH - obs distance (unit : M)
inc = 80            # inclination
FOV = 10            # Field of View (FOV) (unit : deg)
PPD = 60            # Pixel per Deg. (PPD)

r_screen *= M

# ------------------------------ Main ------------------------------

ax = None
BH = Schwarzchild(M=M)

x_screen = np.array([r_screen, inc*np.pi/180, 0])
freq = 1.3
BH.screenInitSetting(x_screen, freq=freq, FOV=FOV, PPD=PPD)

r_in = 6*M
r_out = 20*M
img = BH.rayTracerBatch(mode="thin_disk", r_in=r_in, r_out=r_out)
#z_img = img[:, :, 0]
#intensity_img = img[:, :, 1]

np.save(SAVE_PATH+"/BH_raw_img.npy", img)

ax = plot2DSchwarzchild(img,
                        mode=None,
                        FOV=FOV, PPD=PPD, r_screen=r_screen,
                        M=M, inc=inc)
#plt.show()
plt.savefig(SAVE_PATH+"/BH_img.png", dpi=200)

ax = plot2DSchwarzchild(img,
                        mode='withGuide',
                        FOV=FOV, PPD=PPD, r_screen=r_screen,
                        M=M, inc=inc)
#plt.show()
plt.savefig(SAVE_PATH+"/BH_img_guide.png", dpi=200)




raise

N = 15
P_screen = BH.photonScreen()
P_screen_flatten = P_screen.reshape(4, -1)
sampling_idx = np.random.choice(np.arange(P_screen_flatten.shape[1]), N, replace=False)
init_mom_arr = P_screen_flatten[:, sampling_idx]

for i in tqdm(range(N)):
    init_pos = np.array([0, x_screen[0], x_screen[1], x_screen[2]])
    init_mom = init_mom_arr[:, i]
    y_init = np.concatenate((init_pos, init_mom), axis=0)

    geodesic = lambda l, y: BH.geodesic(l, y)
    terminalCondition = lambda y, h: BH.terminalCondition(y, h)
    _, Y, _ = RK4(geodesic, t_range=(0, 500), initial=y_init, h=1e-1, terminalCondition=terminalCondition)
    x_sph = Y[:, 1:4]

    freq = init_mom[0]
    ax = plot3DSchwarzchild(x_sph, ax=ax, M=M, freq=freq, lim_set=(-r_screen, r_screen))
plt.show()