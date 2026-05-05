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

def plot2DSchwarzchild(img, **kwargs):
    FOV = kwargs['FOV']
    PPD = kwargs['PPD']
    r_screen = kwargs['r_screen']
    M = kwargs['M']

    cmap = plt.cm.hot.copy()
    cmap.set_bad(color='black')

    ax = plt.figure()
    plt.imshow(img, extent = [-FOV/2, FOV/2, -FOV/2, FOV/2], cmap=cmap, vmin=0.5, vmax=1.5)

    theta = np.linspace(0, 2*np.pi, 100)
    r_shadow = (np.sqrt(27)*M/r_screen) * 180/np.pi
    X = r_shadow * np.cos(theta)
    Y = r_shadow * np.sin(theta)
    plt.plot(X, Y, color='red', linestyle='dashed', alpha=0.7, label="Theoretical Shadow")

    plt.title("Schwarzchild BH Shadow")
    plt.xlabel("deg")
    plt.ylabel("deg")
    plt.legend()
    plt.colorbar(label = r'redshift factor $g=\nu_{\rm obs}/\nu_{\rm emit}$')
    #cbar.set_label('photon fate')
    #cbar.set_ticks([0, 0.5, 1])
    #cbar.set_ticklabels(['BH', 'Disk', 'Escape'])
    return ax

def plot3DSchwarzchild(x_sph, ax, **kwargs):
    """
    ## Plot the 3D Graphics around the Schwarzchild BH
    #### [Parameter]
    x_sph : spherical coordinates
    - size = (t_len, 3)
    - x_sph = (r, theta, phi)
    """
    r = x_sph[:, 0]
    theta = x_sph[:, 1]
    phi = x_sph[:, 2]
    X, Y, Z = sph2Cart(r, theta, phi)

    M = kwargs['M']         # BH Mas
    freq = kwargs['freq']   # observed frequency of photon
    lim_set = kwargs['lim_set']

    # Basic Setting
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=0.1)
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter((0), (0), (0), color='black', s=50) # Plot the BH

        # Event Horizon
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 20)
        r_hor = 2*M
        x = r_hor * np.outer(np.cos(phi), np.sin(theta))
        y = r_hor * np.outer(np.sin(phi), np.sin(theta))
        z = r_hor * np.outer(np.ones_like(phi), np.cos(theta))
        ax.plot_surface(x, y, z, color='black', edgecolor='none', alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # matplotlib 요구사항

        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label('frequency')

    color = cmap(norm(np.log(freq)))
    #print(freq, np.log(freq))

    ax.plot(X, Y, Z, lw=1, color=color)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(lim_set)
    ax.set_xlabel('X')
    ax.set_ylim(lim_set)
    ax.set_ylabel('Y')
    ax.set_zlim(lim_set)
    ax.set_zlabel('Z')
    ax.set_title("Schwarzchild Geodesic")

    return ax
