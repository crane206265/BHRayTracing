#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################
import numpy as np
import matplotlib.pyplot as plt

from utils import plot2DBH
from AccretionDisk import SimpleDisk
from BlackHole import Schwarzchild, Kerr



#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################
SAVE_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\BHRayTracing\saved_data"


M = 1               # Mass of BH
r_screen = 200       # BH - obs distance (unit : M)
inc = 80            # inclination
FOV = 10            # Field of View (FOV) (unit : deg)
PPD = 60            # Pixel per Deg. (PPD)

r_screen *= M


ax = None
Disk = SimpleDisk(q=2)
#BH = Schwarzchild(M=M, Disk=Disk)
BH = Kerr(M=M, J=0.8*M, Disk=Disk)

#######################################################################
# ------------------------------- Main --------------------------------
#######################################################################
img = np.load(SAVE_PATH+"/Kerr_BH_raw_img.npy")
ax = plot2DBH(img, BH=BH,
              mode=None,
              FOV=FOV, PPD=PPD,
              r_screen=r_screen, inc=inc)
plt.show()
#plt.savefig(SAVE_PATH+"/Kerr_BH_img.png", dpi=200)

raise
ax = plot2DBH(img,
              mode='withGuide',
              FOV=FOV, PPD=PPD, r_screen=r_screen,
              M=M, inc=inc)
plt.show()
#plt.savefig(SAVE_PATH+"/BH_img_guide.png", dpi=200)