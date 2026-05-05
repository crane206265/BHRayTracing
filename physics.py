# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ Functions ------------------------------
def geodesicSchwarzchild(l, y, M):
    """
    ## Geodesic Equation of Schwarzchild Metric
    - form for RK4
    #### [Parameter]
    l : affine parameter ($\lambda$) \\
    y : Input. ($t, r, \theta, \phi, P^t, P^r, P^{\theta}, P^{\phi}$)
    """
    t, r, theta, phi, Pt, Pr, Ptheta, Pphi = y
    eps = 1e-8

    dt = Pt
    dr = Pr
    dtheta = Ptheta
    dphi = Pphi
    dPt = -(2*M/(r*(r-2*M)))*Pr*Pt
    dPr = -(M/(r**3))*(r-2*M)*Pt*Pt + (M/(r*(r-2*M)))*Pr*Pr + (r-2*M)*(Ptheta*Ptheta + (np.sin(theta)*Pphi)**2)
    dPtheta = -(2/r)*Ptheta*Pr + np.sin(theta)*np.cos(theta)*Pphi*Pphi
    dPphi = -(2/r)*Pphi*Pr - (2*np.cos(theta)/(np.sin(theta)+eps))*Ptheta*Pphi
    return np.array([dt, dr, dtheta, dphi, dPt, dPr, dPtheta, dPphi])

def terminalConditionSchwarzchild(y, h, **kwargs):
    """
    #### Terminal Conditions for Schwarzchild BH
    - returns boolean
    #### [Paramter]
    y : coordinate, momentum\\
    h : step size of RK4\\
    """
    M = kwargs['M']

    # NaN Detection
    if np.isnan(y).any():
        msg = "NaN Detected"
        return True, msg
    # INF Detection
    if np.isinf(y).any():
        msg = "INF Detected"
        return True, msg
    # Particle went into the Event Horizon
    if y[1] < 2*M + h:
        msg = "Particle went into the event horizon"
        return True, msg
    msg = "None"
    return False, msg