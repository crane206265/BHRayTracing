from sympy import symbols, sin, Matrix
from sympy.diffgeom import Manifold, Patch, CoordSystem, metric_to_Christoffel_2nd
from sympy.diffgeom import TensorProduct as TP


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