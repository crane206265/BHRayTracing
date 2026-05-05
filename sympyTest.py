from sympy import symbols, sin, Matrix
from sympy.diffgeom import Manifold, Patch, CoordSystem, metric_to_Christoffel_2nd
from sympy.diffgeom import TensorProduct as TP

"""
G, c = 1, 1
M = 1

m = Manifold('M', 4)
p = Patch('P', m)

sch = CoordSystem('Sch', p, ['t', 'r', 'theta', 'phi'])
t, r, theta, phi = sch.coord_functions()
dt, dr, dtheta, dphi = sch.base_oneforms()

Rs = 2*G*M/(c**2)
metric = -(1-Rs/r)*TP(dt, dt) + (1/(1-Rs/r))*TP(dr, dr) + (r**2)*TP(dtheta, dtheta) + ((r*sin(theta))**2)*TP(dphi, dphi)


print(metric_to_Christoffel_2nd(metric))
"""



import numpy as np
def photonScreenPoint(alpha, beta):
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
    print(mom4_local.shape)
    scale = 2/mom4_local[0]
    return scale*mom4_local

alpha = np.arange(1, 10)
print(photonScreenPoint(alpha, alpha))


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