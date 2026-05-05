# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ Functions ------------------------------
def sph2Cart(r, theta, phi):
    """
    ## Coordinate Transformation : Spherical -> Cartesian
    #### [Parameter]
    r : radial coord.\\
    theta : polar angle\\
    phi : azimuthal angle\\
    """
    X = r*np.sin(theta)*np.cos(phi)
    Y = r*np.sin(theta)*np.sin(phi)
    Z = r*np.cos(theta)
    return X, Y, Z

def plot3D(x_sph, ax, **kwargs):
    """
    ## Plot the 3D Graphics around the Black Hole
    #### [Parameter]
    x_sph : spherical coordinates
    - size = (t_len, 3)
    - x_sph = (r, theta, phi)
    """
    r = x_sph[:, 0]
    theta = x_sph[:, 1]
    phi = x_sph[:, 2]
    X, Y, Z = sph2Cart(r, theta, phi)

    M = kwargs['M']
    
    # Basic Setting
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter((0), (0), (0), color='black', s=50) # Plot the BH

        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 20)
        r_hor = 2*M
        x = r_hor * np.outer(np.cos(phi), np.sin(theta))
        y = r_hor * np.outer(np.sin(phi), np.sin(theta))
        z = r_hor * np.outer(np.ones_like(phi), np.cos(theta))
        ax.plot_surface(x, y, z, color='black', edgecolor='none', alpha=0.5)

    ax.plot(X, Y, Z, lw=1)
    lim_set = (-10, 10)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(lim_set)
    ax.set_xlabel('X')
    ax.set_ylim(lim_set)
    ax.set_ylabel('Y')
    ax.set_zlim(lim_set)
    ax.set_zlabel('Z')
    ax.set_title("Schwarzchild Geodesic")

    return ax