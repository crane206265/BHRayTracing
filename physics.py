# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ Functions ------------------------------
class Schwarzchild():
    """
    # Schwarzchild Black Hole
    - free parameter : $M$ (mass)
    #### [Helper Method]
    - geodesic
    - terminalCondition
    - photon_tMomentum
    """
    def __init__(self, M):
        self.M = M

    def geodesic(self, l, y):
        """
        ## Geodesic Equation of Schwarzchild Metric
        - form for RK4
        #### [Parameter]
        l : affine parameter ($\lambda$) \\
        y : Input. ($t, r, \theta, \phi, P^t, P^r, P^{\theta}, P^{\phi}$)
        """
        t, r, theta, phi, Pt, Pr, Ptheta, Pphi = y
        M = self.M
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

    def terminalCondition(self, y, h, **kwargs):
        """
        #### Terminal Conditions for Schwarzchild BH
        - returns boolean
        #### [Paramter]
        y : coordinate, momentum\\
        h : step size of RK4\\
        """
        M = self.M

        # NaN Detection
        if np.isnan(y).any():
            endType = "Error"
            msg = "NaN Detected"
            return True, msg
        # INF Detection
        if np.isinf(y).any():
            endType = "Error"
            msg = "INF Detected"
            return True, msg
        
        # Particle went into the Event Horizon
        if y[1] < 2*M + h:
            endType = "EventHorizon"
            msg = "Particle went into the event horizon"
            return True, msg
        msg = "None"
        return False, msg

    def photon_tMomentum(self, x_sph, p_sph, **kwargs):
        """
        #### Photon Momentum with Schwarzchild Metric
        - return the 4-momentum corr. given 3-momentum of photon
        - using null condition
        #### [Parameter]
        x_sph : 3-coord.
        p_sph_3 : 3-momentum
        """
        r, theta, phi = x_sph
        Pr, Ptheta, Pphi = p_sph
        M = self.M

        f = 1-2*M/r
        Pt = np.sqrt(f*(Pr*Pr/f + (r*Ptheta)**2 + (r*np.sin(theta)*Pphi)**2))
        return np.array([Pt, Pr, Ptheta, Pphi])
    
    def vierbein(self, x_sph):
        """
        #### Vierbein
        - global -> local
        """
        r, theta, phi = x_sph
        M = self.M

        e = np.zeros((4, 4))
        e[0, 0] = np.sqrt(1-2*M/r)
        e[1, 1] = 1/np.sqrt(1-2*M/r)
        e[2, 2] = r
        e[3, 3] = r*np.sin(theta)
        return e

    # -------------------- Screen Setting --------------------

    def screenInitSetting(self, x_screen, freq, FOV, PPD):
        """
        #### Initial Setting of Screen
        #### [Paramter]
        x_screen : position of screen with spherical coordinates
        freq : observing frequency
        FOV : field of view (FOV) (unit : deg)
        PPD : pixel per degree (PPD) (unit : #/deg)
        """
        self.x_screen = x_screen
        self.freq = freq
        self.FOV = FOV
        self.PPD = PPD
        self.img = np.zeros((FOV*PPD, FOV*PPD))

    def photonScreenPoint(self, alpha, beta):
        """
        #### Photon Mometum at Screen Point
        - return the 4-mometum of photon starting at given point on the screen
        #### [Parameter]
        alpha : screen x-coord.
        beta : screen y-coord.
        """
        X = np.tan(alpha)
        Y = np.tan(beta)

        # mom4_local : 4-momentum of local frame (Minkowski)
        mom4_local = np.array([np.ones_like(X),
                               -1/np.sqrt(1+X**2+Y**2),
                               Y/np.sqrt(1+X**2+Y**2),
                               X/np.sqrt(1+X**2+Y**2)])
        mom4_local = photonFreq(mom4_local, self.freq)

        # mom4_local : 4-momentum of global frame (Schwarzchild)
        e = self.vierbein(self.x_screen)   # (4,4)
        e_inv = np.linalg.inv(e)           # (4,4)
        mom4_global = np.einsum('ij,j...->i...', e_inv, mom4_local) # (4,N)
        return mom4_global

    def photonScreen(self):
        alpha = np.linspace(-self.FOV/2, self.FOV/2, self.FOV*self.PPD)
        beta = np.linspace(-self.FOV/2, self.FOV/2, self.FOV*self.PPD)
        alpha = np.tile(alpha, self.FOV*self.PPD).reshape((self.FOV*self.PPD, self.FOV*self.PPD))
        beta = np.repeat(beta, self.FOV*self.PPD).reshape((self.FOV*self.PPD, self.FOV*self.PPD))

        P_screen = self.photonScreenPoint(alpha, beta)
        return P_screen

    def screenSampling(self, N):
        """
        #### Sampling the photon on screen
        - random sampling $N$ 4-momentums of photons from the screen
        #### [Parameter]
        N : # of photon to sampling
        """
        P_screen = self.photonScreen()
        P_screen_flatten = P_screen.reshape(4, -1)
        sampling_idx = np.random.choice(np.arange(P_screen_flatten.shape[1]), N, replace=False)
        return P_screen_flatten[:, sampling_idx]



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