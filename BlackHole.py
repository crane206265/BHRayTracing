#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################

import numpy as np
from tqdm import tqdm

from numerical import RK4, RK4Batch
from physics import BlackHole, AccDisk
import physics as PH



#######################################################################
# ------------------------------ BH Class -----------------------------
#######################################################################

class Schwarzchild(BlackHole):
    """
    # Schwarzchild Black Hole
    - free parameter : $M$ (mass)
    #### [Helper Method]
    - geodesic
    - terminalCondition
    - photon_tMomentum
    """
    def __init__(self, M, Disk:AccDisk):
        super.__init__(M=M, J=None, Q=None)
        self.Disk = Disk
        self.Disk.setBH(self)

    # -------------------- Basic Physics --------------------
    def metricTensor(self, x):
        """
        #### Metric Tensor
        $$g_{\mu\nu}$$
        """
        t, r, theta, phi = x
        M = self.M
        N = x.shape[1]

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
    
    def vierbein(self, x):
        """
        #### Vierbein
        - global -> local
        """
        t, r, theta, phi = x
        M = self.M

        e = np.zeros((4, 4))
        e[0, 0] = np.sqrt(1-2*M/r)
        e[1, 1] = 1/np.sqrt(1-2*M/r)
        e[2, 2] = r
        e[3, 3] = r*np.sin(theta)
        return e

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

    # -------------------- Initial / Final Conditions --------------------
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
            freq_emit = self.Disk.diskEmitFreq(y_on[:4], y_on[4:])
            intensity_emit = self.Disk.diskIntensity(y_on[:4], q=2)
            disk_info = np.stack((freq_emit, intensity_emit), axis=1)

        terminated = np.logical_or(endType[:, 0], endType[:, 1])
        terminated = np.logical_or(terminated, endType[:, 2])
        return terminated, endType, disk_info

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
        self.img = np.full((self.N, self.N, 2), np.nan) # idx 0: z / idx 1: Intensity

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
        mom4_local = PH.photonFreq(mom4_local, self.freq)

        # mom4_local : 4-momentum of global frame (Schwarzchild)
        e = self.vierbein(np.insert(self.x_screen, 0, 0))   # (4,4)
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
    
    # -------------------- Ray Tracing --------------------
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
            _, _, endType, disk_info = RK4Batch(geodesic, t_range=(0, 500), initial=y_init_batch, h=2e-2, terminalCondition=terminalCondition)

            freq_emit = disk_info[:, 0]
            intensity_emit = disk_info[:, 1]
            for local_idx, global_idx in tqdm(enumerate(range(batch_start, batch_end))):
                i = global_idx//self.N
                j = global_idx%self.N
                if endType[local_idx, 0]      : self.img[i, j, :] = np.full((self.img.shape[-1]), np.nan)
                elif endType[local_idx, 1]    : self.img[i, j, :] = np.full((self.img.shape[-1]), np.nan)
                elif endType[local_idx, 2]    :
                    g = self.freq/freq_emit[local_idx]
                    self.img[i, j, 0] = g
                    self.img[i, j, 1] = (g**3)*intensity_emit[local_idx]
        return self.img



class Kerr(BlackHole):
    """
    # Kerr Black Hole
    - free parameter : $M$ (mass), $J$ (Spin Angular Momentum)
    #### [Helper Method]
    - geodesic
    - terminalCondition
    - photon_tMomentum
    """
    def __init__(self, M, J, Disk:AccDisk):
        super.__init__(M=M, J=J, Q=None)
        self.Disk = Disk
        self.Disk.setBH(self)

    # -------------------- Basic Physics --------------------
    def metricTensor(self, x):
        """
        #### Metric Tensor
        $$g_{\mu\nu}$$
        """
        t, r, theta, phi = x
        M = self.M
        J = self.J
        N = x.shape[1]

        rs = 2*M # Schwarzschild radius
        a = J/M # Kerr param.

        D = r*r - rs*r + a*a
        S = r*r + a*a*np.cos(theta)*np.cos(theta)
        gtt = -(1 - (rs*r)/(S))
        gtphi = -(rs*r*a*np.sin(theta)*np.sin(theta))/(S)
        gphiphi = (r*r + a*a + (rs*r*a*a*np.sin(theta)*np.sin(theta))/(S))*np.sin(theta)*np.sin(theta)
        if N == 1:
            g = np.zeros((4, 4))
            g[0, 0] = gtt
            g[1, 1] = S/D
            g[2, 2] = S
            g[3, 3] = gphiphi
            g[0, 3] = gtphi
            g[3, 0] = gtphi
            
        if N > 1:
            g = np.zeros((4, 4, N))
            g[0, 0, :] = gtt
            g[1, 1, :] = S/D
            g[2, 2, :] = S
            g[3, 3, :] = gphiphi
            g[0, 3, :] = gtphi
            g[3, 0, :] = gtphi
        
        return g
    
    def Christoffel(self, x):
        t, r, theta, phi = x
        M = self.M
        J = self.J
        N = x.shape[1]

        rs = 2*M # Schwarzschild radius
        a = J/M # Kerr param.

        # for convention, define trigonometric values and square of var.
        r2 = r**2
        a2 = a**2
        sinth = np.sin(theta)
        costh = np.cos(theta)
        sinth2 = sinth**2
        costh2 = costh**2


        D = r2 - 2*M*r + a2
        S = r2 + a2*costh2
        A = (r2+a2)**2 - a2*D*sinth2
        
        # for convention, define the frequent terms
        S2 = S**2
        Sm = r2 - a2*costh2 # looks like Sigma, but minus
        sincosth = sinth*costh

        Gamma = np.zeros((4, 4, 4, N)) # idx : \mu, \alpha, \beta
        # N==1 case removed
        # ${\Gamma^\mu}_{\alpha\beta}$
        # Symmetric matrix for fixed \mu --> Gamma = A + A.T
        # thus, to generate A, diagonals will assigned as half
        # 0. \mu = t
        Gamma[0, 0, 0, :] = 0 / 2                                                          # t,t,t         | diag
        Gamma[0, 0, 1, :] = (rs*(r2+a2)*Sm) / (2*S2*D)                                     # t,t,r
        Gamma[0, 0, 2, :] = -(rs*a2*r*sincosth) / (S2)                                     # t,t,theta
        Gamma[0, 0, 3, :] = 0                                                              # t,t,phi
        Gamma[0, 1, 1, :] = 0 / 2                                                          # t,r,r         | diag
        Gamma[0, 1, 2, :] = 0                                                              # t,r,theta
        Gamma[0, 1, 3, :] = (rs*a*sinth2*(a2*costh2*(a2-r2)-r2*(a2+3*r2))) / (2*S2*D)      # t,r,phi
        Gamma[0, 2, 2, :] = 0 / 2                                                          # t,theta,theta | diag
        Gamma[0, 2, 3, :] = (rs*a2*a*r*sinth2*sincosth) / (S2)                             # t,theta,phi
        Gamma[0, 3, 3, :] = 0 / 2                                                          # t,phi,phi     | diag

        # 1. \mu = r
        Gamma[1, 0, 0, :] = ((rs*D*Sm) / (2*S2*S)) / 2                                     # r,t,t         | diag
        Gamma[1, 0, 1, :] = 0                                                              # r,t,r
        Gamma[1, 0, 2, :] = 0                                                              # r,t,theta
        Gamma[1, 0, 3, :] = -(D*rs*a*sinth2*Sm) / (2*S2*S)                                 # r,t,phi
        Gamma[1, 1, 1, :] = ((2*r*a2*sinth2-rs*Sm) / (2*S*D)) / 2                          # r,r,r         | diag
        Gamma[1, 1, 2, :] = -(a2*sincosth) / (S)                                           # r,r,theta
        Gamma[1, 1, 3, :] = 0                                                              # r,r,phi
        Gamma[1, 2, 2, :] = -((r*D) / (S)) / 2                                             # r,theta,theta | diag
        Gamma[1, 2, 3, :] = 0                                                              # r,theta,phi
        Gamma[1, 3, 3, :] = ((D*sinth2*(-2*r*S*S+rs*a2*sinth2*Sm)) / (2*S2*S)) / 2         # r,phi,phi     | diag

        # 2. \mu = \theta
        Gamma[2, 0, 0, :] = -((rs*a2*r*sincosth) / (S2*S)) / 2                             # theta,t,t         | diag
        Gamma[2, 0, 1, :] = 0                                                              # theta,t,r
        Gamma[2, 0, 2, :] = 0                                                              # theta,t,theta
        Gamma[2, 0, 3, :] = (rs*a*r*(r2+a2)*sincosth) / (S2*S)                             # theta,t,phi
        Gamma[2, 1, 1, :] = ((a2*sincosth) / (S*D)) / 2                                    # theta,r,r         | diag
        Gamma[2, 1, 2, :] = r / S                                                          # theta,r,theta
        Gamma[2, 1, 3, :] = 0                                                              # theta,r,phi
        Gamma[2, 2, 2, :] = -((a2*sincosth) / (S)) / 2                                     # theta,theta,theta | diag
        Gamma[2, 2, 3, :] = 0                                                              # theta,theta,phi
        Gamma[2, 3, 3, :] = -((sincosth*(A*S+(r2+a2)*rs*a2*r*sinth2)) / (S2*S)) / 2        # theta,phi,phi     | diag
            
        # 3. \mu = \phi
        Gamma[3, 0, 0, :] = 0 / 2                                                          # phi,t,t         | diag
        Gamma[3, 0, 1, :] = (rs*a*Sm) / (2*S2*D)                                           # phi,t,r
        Gamma[3, 0, 2, :] = -(rs*a*r) / (S2*np.tan(theta))                                 # phi,t,theta
        Gamma[3, 0, 3, :] = 0                                                              # phi,t,phi
        Gamma[3, 1, 1, :] = 0 / 2                                                          # phi,r,r         | diag
        Gamma[3, 1, 2, :] = 0                                                              # phi,r,theta
        Gamma[3, 1, 3, :] = (2*r*S2 + rs*(a2*a2*sinth2*costh2-r2*(S+r2+a2))) / (2*S2*D)    # phi,r,phi
        Gamma[3, 2, 2, :] = 0 / 2                                                          # phi,theta,theta | diag
        Gamma[3, 2, 3, :] = (S2 + rs*a2*r*sinth2) / (S2*np.tan(theta))                     # phi,theta,phi
        Gamma[3, 3, 3, :] = 0 / 2                                                          # phi,phi,phi     | diag
            
        Gamma = Gamma + np.transpose(Gamma, (0, 2, 1, 3))

        return Gamma

    def vierbein(self, x):
        """
        #### Vierbein
        - global -> local
        """
        t, r, theta, phi = x
        M = self.M
        J = self.J

        e = np.zeros((4, 4))
        e[0, 0] = np.sqrt(1-2*M/r)
        e[1, 1] = 1/np.sqrt(1-2*M/r)
        e[2, 2] = r
        e[3, 3] = r*np.sin(theta)
        return e

    def geodesic(self, l, y):
        """
        ## Geodesic Equation of Schwarzchild Metric
        - form for RK4
        #### [Parameter]
        l : affine parameter ($\lambda$) \\
        y : Input. ($t, r, \theta, \phi, P^t, P^r, P^{\theta}, P^{\phi}$)
        """
        x = y[0:4] #(4, N)
        p = y[4:8] #(4, N)
        M = self.M
        eps = 1e-8
        N = y.shape[1]

        Gamma = self.Christoffel(x) # (4, 4, 4, N)
        result = np.zeros((8, N))
        
        result[0:4] = p # (4, N)
        result[4:8] = -np.einsum('ijkn,jn,kn->in', Gamma, p, p) # (4,4,N)
        
        return result

    # -------------------- Initial / Final Conditions --------------------
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
            freq_emit = self.Disk.diskEmitFreq(y_on[:4], y_on[4:])
            intensity_emit = self.Disk.diskIntensity(y_on[:4], q=2)
            disk_info = np.stack((freq_emit, intensity_emit), axis=1)

        terminated = np.logical_or(endType[:, 0], endType[:, 1])
        terminated = np.logical_or(terminated, endType[:, 2])
        return terminated, endType, disk_info

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
        self.img = np.full((self.N, self.N, 2), np.nan) # idx 0: z / idx 1: Intensity

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
        mom4_local = PH.photonFreq(mom4_local, self.freq)

        # mom4_local : 4-momentum of global frame (Schwarzchild)
        e = self.vierbein(np.insert(self.x_screen, 0, 0))   # (4,4)
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
    
    # -------------------- Ray Tracing --------------------
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
            _, _, endType, disk_info = RK4Batch(geodesic, t_range=(0, 500), initial=y_init_batch, h=2e-2, terminalCondition=terminalCondition)

            freq_emit = disk_info[:, 0]
            intensity_emit = disk_info[:, 1]
            for local_idx, global_idx in tqdm(enumerate(range(batch_start, batch_end))):
                i = global_idx//self.N
                j = global_idx%self.N
                if endType[local_idx, 0]      : self.img[i, j, :] = np.full((self.img.shape[-1]), np.nan)
                elif endType[local_idx, 1]    : self.img[i, j, :] = np.full((self.img.shape[-1]), np.nan)
                elif endType[local_idx, 2]    :
                    g = self.freq/freq_emit[local_idx]
                    self.img[i, j, 0] = g
                    self.img[i, j, 1] = (g**3)*intensity_emit[local_idx]
        return self.img
