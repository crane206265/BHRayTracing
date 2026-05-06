# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt

from utils import plot2DSchwarzchild


# --------------------------- Param. Setting --------------------------
SAVE_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\BHRayTracing\saved_data"


M = 1               # Mass of BH
r_screen = 200       # BH - obs distance (unit : M)
inc = 80            # inclination
FOV = 10            # Field of View (FOV) (unit : deg)
PPD = 60            # Pixel per Deg. (PPD)

r_screen *= M

# ------------------------------ Main ------------------------------
img = np.load(SAVE_PATH+"/BH_raw_img.npy")

ax = plot2DSchwarzchild(img,
                        mode=None,
                        FOV=FOV, PPD=PPD, r_screen=r_screen,
                        M=M, inc=inc)
plt.show()
#plt.savefig(SAVE_PATH+"/BH_img.png", dpi=200)

ax = plot2DSchwarzchild(img,
                        mode='withGuide',
                        FOV=FOV, PPD=PPD, r_screen=r_screen,
                        M=M, inc=inc)
plt.show()
#plt.savefig(SAVE_PATH+"/BH_img_guide.png", dpi=200)