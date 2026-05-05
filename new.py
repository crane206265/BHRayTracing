# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt



# ------------------------------ Param.s ------------------------------

M = 1               # Mass of BH
r_screen = 50       # BH - obs distance (unit : M)
FOV = 20            # Field of View (FOV) (unit : deg)
PPD = 10            # Pixel per Deg. (PPD)


# --------------------------- Param. Setting --------------------------
r_screen *= M
img = np.zeros((FOV*PPD, FOV*PPD))



# ------------------------------ Methods ------------------------------
def RK4(f, initial, t_range, h=1e-3):
    """
    ## Runge-Kutta Method of 4th Order
    $$ y' = f(t,y), y(t_0) = y_0 $$ \\
    $$ y_{n+1} = y_n + \frac{1}{6}h(k_1 + 2k_2 + 2k_3 + k_4) $$\\
    $$ t_{n+1} = t_n + h $$
    """
    t_arr = np.arange(t_range[0], t_range[1], h)
    t_len = t_arr.shape[0]

    y_dim = initial.shape[0]
    y_arr = np.zeros((t_len, y_dim))
    y_arr[0] = initial

    for i in range(t_len):
        t_i = t_arr[i]
        y_i = y_arr[i]

        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5*h, y_i + 0.5*h*k1)
        k3 = f(t_i + 0.5*h, y_i + 0.5*h*k2)
        k4 = f(t_i + h, y_i + h*k3)

        if i < t_len-1:
            y_arr[i+1] = y_i + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return t_arr, y_arr

def geodesicSch(l, y, M):
    """
    ## Geodesic Equation of Schwarzchild Metric
    - form for RK4
    #### [Parameter]
    l : affine parameter ($\lambda$) \\
    y : Input. ($t, r, \theta, \phi, P^t, P^r, P^{\theta}, P^{\phi}$)
    """
    t, r, theta, phi, Pt, Pr, Ptheta, Pphi = y
    dt = Pt
    dr = Pr
    dtheta = Ptheta
    dphi = Pphi
    dPt = -(2*M/(r*(r-2*M)))*Pr*Pt
    dPr = -(M/(r**3))*(r-2*M)*Pt*Pt + (M/(r*(r-2*M)))*Pr*Pr + (r-2*M)*(Ptheta*Ptheta + (np.sin(theta)*Pphi)**2)
    dPtheta = -(2/r)*Ptheta*Pr + np.sin(theta)*np.cos(theta)*Pphi*Pphi
    dPphi = -(2/r)*Pphi*Pr - (2*np.cos(theta)/np.sin(theta))*Ptheta*Pphi
    return np.array([dt, dr, dtheta, dphi, dPt, dPr, dPtheta, dPphi])

