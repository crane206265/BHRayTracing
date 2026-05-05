# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from numerical import RK4, RK4Batch


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

    def terminalCondition(self, y, y1, h, mode=None, **kwargs):
        """
        #### Terminal Conditions for Schwarzchild BH
        - returns boolean
        #### [Paramter]
        y : coordinate, momentum at t=i\\
        y1 : coordinate, momentum at t=i-1\\
        h : step size of RK4\\
        """
        M = self.M
        r_max = kwargs['r_max']

        # NaN Detection
        if np.isnan(y).any():
            endType = "Error"
            msg = "NaN Detected"
            return True, msg, endType
        # INF Detection
        if np.isinf(y).any():
            endType = "Error"
            msg = "INF Detected"
            return True, msg, endType
        
        # Particle went into the Event Horizon
        if y[1] < 2*M + h:
            endType = "EventHorizon"
            msg = "Particle went into the event horizon"
            return True, msg, endType
        
        # Particle escaped the BH
        if y[1] > r_max + 1*M:
            endType = "Escape"
            msg = "Particle escaped the BH"
            return True, msg, endType
        
        # thin disk
        if mode == "thin_disk":
            r_in = kwargs['r_in']
            r_out = kwargs['r_out']
            crossing = ((y[2]-np.pi/2) * (y1[2]-np.pi/2) < 0)

            w = np.abs(y1[2] - np.pi/2) / (np.abs(y1[2] - np.pi/2) + np.abs(y[2] - np.pi/2))
            r_on = (1-w)*y1[1] + w*y[1]
            on_disk = (r_on > r_in) and (r_on < r_out)
            if crossing and on_disk:
                endType = "Disk"
                msg = "Particle crossed the disk"
                return True, msg, endType

        msg = "None"
        return False, msg, None

    def terminalConditionBatch(self, y, y1, h, mode=None, **kwargs):
        """
        #### Terminal Conditions for Schwarzchild BH
        - returns boolean
        #### [Paramter]
        y : coordinate, momentum at t=i\\
        y1 : coordinate, momentum at t=i-1\\
        h : step size of RK4\\
        """
        M = self.M
        r_max = kwargs['r_max']
        
        endType = np.full((y.shape[1], 3), False)
        # Particle went into the Event Horizon
        endType[:, 0] = y[1] < 2*M + h
        
        # Particle escaped the BH
        endType[:, 1] = y[1] > r_max + 1*M
        
        # thin disk
        if mode == "thin_disk":
            r_in = kwargs['r_in']
            r_out = kwargs['r_out']
            tol = 1e-2
            eps = 1e-8

            crossing = ((y[2]-np.pi/2) * (y1[2]-np.pi/2) < 0)
            crossing = crossing | (np.abs((y[2]+y1[2])/2 - np.pi/2) < tol)

            w = np.abs(y1[2] - np.pi/2) / ((np.abs(y1[2] - np.pi/2) + np.abs(y[2] - np.pi/2)) + eps)
            y_on = (1-w)*y1 + w*y
            r_on = y_on[1]
            on_disk = (r_on > r_in) & (r_on < r_out)
            endType[:, 2] = crossing & on_disk
            freq_emit = self.diskEmitFreq(y_on[:4], y_on[4:])

        terminated = np.logical_or(endType[:, 0], endType[:, 1])
        terminated = np.logical_or(terminated, endType[:, 2])
        return terminated, endType, freq_emit

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
    
    def metricTensor(self, x_sph):
        """
        #### Metric Tensor
        $$g_{\mu\nu}$$
        """
        r, theta, phi = x_sph
        M = self.M
        N = x_sph.shape[1]

        if N == 1:
            g = np.zeros((4, 4))
            g[0, 0] = -(1-2*M/r)
            g[1, 1] = 1/(1-2*M/r)
            g[2, 2] = r**2
            g[3, 3] = (r*np.sin(theta))**2
            return g
        if N > 1:
            g = np.zeros((4, 4, N))
            g[0, 0, :] = -(1-2*M/r)
            g[1, 1, :] = 1/(1-2*M/r)
            g[2, 2, :] = r**2
            g[3, 3, :] = (r*np.sin(theta))**2
            return g
    
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
        Ut = 1/np.sqrt(1-3*self.M/r)
        Uphi = np.sqrt(self.M/(r**3))*Ut
        U_disk = np.array([Ut,
                           np.zeros_like(Ut),
                           np.zeros_like(Ut),
                           Uphi])
        # u_disk : (4, N)
        # p : (4, N)
        metric = self.metricTensor(x[1:]) # (4,4,N)

        U_disk_covar = np.einsum('ij...,j...->i...', metric, U_disk) # (4, N)
        freq_emit = -np.einsum('i...,i...->...', p, U_disk_covar) #(N)
        return freq_emit

    # -------------------- Screen Setting --------------------

    def screenInitSetting(self, x_screen, freq, FOV, PPD):
        """
        #### Initial Setting of Screen
        #### [Paramter]
        x_screen : 3-position of screen with spherical coordinates
        freq : observing frequency
        FOV : field of view (FOV) (unit : deg)
        PPD : pixel per degree (PPD) (unit : #/deg)
        """
        self.x_screen = x_screen
        self.freq = freq
        self.FOV = FOV
        self.PPD = PPD

        self.init_pos = np.array([0, self.x_screen[0], self.x_screen[1], self.x_screen[2]])
        self.N = int(FOV*PPD)
        self.img = np.zeros((self.N, self.N))

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
        alpha = np.linspace(-self.FOV/2, self.FOV/2, self.N) * np.pi/180
        beta = np.linspace(-self.FOV/2, self.FOV/2, self.N) * np.pi/180
        alpha = np.tile(alpha, self.N).reshape((self.N, self.N))
        beta = np.repeat(beta, self.N).reshape((self.N, self.N))

        P_screen = self.photonScreenPoint(alpha, beta)
        return P_screen
    
    def rayTracer(self, mode=None, **kwargs):
        if mode == "thin_disk":
            r_in = kwargs['r_in']
            r_out = kwargs['r_out']

        P_screen = self.photonScreen()
        for idx in tqdm(range(self.N*self.N)):
            i = idx//self.N
            j = idx%self.N
            init_mom = P_screen[:, i, j]
            y_init = np.concatenate((self.init_pos, init_mom), axis=0)

            geodesic = lambda l, y: self.geodesic(l, y)
            terminalCondition = lambda y, y1, h: self.terminalCondition(y, y1, h,
                                                                        mode=mode,
                                                                        r_max=self.x_screen[0],
                                                                        r_in=r_in,
                                                                        r_out=r_out)
            _, Y, endType = RK4(geodesic, t_range=(0, 500), initial=y_init, h=1e-1, terminalCondition=terminalCondition)

            if endType == "Error"           : self.img[i, j] = np.nan
            elif endType == "EventHorizon"  : self.img[i, j] = 0
            elif endType == "Escape"        : self.img[i, j] = 1
            elif endType == "Disk"          : self.img[i, j] = 0.5
        return self.img
    
    def rayTracerBatch(self, mode=None, **kwargs):
        if mode == "thin_disk":
            r_in = kwargs['r_in']
            r_out = kwargs['r_out']

        init_pos = np.repeat(self.init_pos[:, np.newaxis], self.N*self.N, axis=1)
        init_mom = self.photonScreen().reshape(4, -1)
        y_init = np.concatenate((init_pos, init_mom), axis=0)

        geodesic = lambda l, y: self.geodesic(l, y)
        terminalCondition = lambda y, y1, h: self.terminalConditionBatch(y, y1, h,
                                                                         mode=mode,
                                                                         r_max=self.x_screen[0],
                                                                         r_in=r_in,
                                                                         r_out=r_out)
        
        batchsize = 5000
        N_total = y_init.shape[1]
        for b in tqdm(range(int(np.ceil(N_total/batchsize)))):
            print(" Batch %d/%d"%(b, int(np.ceil(N_total/batchsize))))
            batch_start = b*batchsize
            batch_end = min((b+1)*batchsize, N_total)
            y_init_batch = y_init[:, batch_start:batch_end]
            _, _, endType, freq_emit = RK4Batch(geodesic, t_range=(0, 500), initial=y_init_batch, h=2e-2, terminalCondition=terminalCondition)

            for local_idx, global_idx in tqdm(enumerate(range(batch_start, batch_end))):
                i = global_idx//self.N
                j = global_idx%self.N
                if endType[local_idx, 0]      : self.img[i, j] = np.nan
                elif endType[local_idx, 1]    : self.img[i, j] = np.nan
                elif endType[local_idx, 2]    : self.img[i, j] = self.freq/freq_emit[local_idx]
        return self.img

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