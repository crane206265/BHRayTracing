#######################################################################
# ------------------------------ Library ------------------------------
#######################################################################
import numpy as np
from tqdm import tqdm


#######################################################################
# ----------------------------- Functions -----------------------------
#######################################################################
def RK4(f, initial, t_range, h=1e-3, terminalCondition=lambda y, h: (True, None)):
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

        y_i1 = y_arr[i-1] if i != 0 else y_i
        terminated, msg, endType = terminalCondition(y_i, y_i1, h=h)
        if terminated:
            #print(" lambda=%5d| "%(i)+msg)
            return t_arr[:i], y_arr[:i, :], endType
        
        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5*h, y_i + 0.5*h*k1)
        k3 = f(t_i + 0.5*h, y_i + 0.5*h*k2)
        k4 = f(t_i + h, y_i + h*k3)

        if i < t_len-1:
            y_arr[i+1] = y_i + h*(k1 + 2*k2 + 2*k3 + k4)/6
    #print("i=%5d| Particle finished the given path"%(i))
    return t_arr, y_arr, endType


def RK4Batch(f, initial, t_range, h=1e-3, terminalCondition=lambda y, h: (True, None), recordHistory=False):
    """
    ## Runge-Kutta Method of 4th Order
    - Batch Parallelized Code
    #### [Formalism]
    $$ y' = f(t,y), y(t_0) = y_0 $$ \\
    $$ y_{n+1} = y_n + \frac{1}{6}h(k_1 + 2k_2 + 2k_3 + k_4) $$\\
    $$ t_{n+1} = t_n + h $$
    """
    t_arr = np.arange(t_range[0], t_range[1], h)
    t_len = t_arr.shape[0]

    y_dim = initial.shape[0]
    N_dim = initial.shape[1]

    # record all history of geodesics
    if recordHistory:
        y_arr = np.zeros((t_len, y_dim, N_dim))
        y_arr[0] = initial
        disk_info = np.zeros((N_dim, 2))

        alive = np.ones(N_dim, dtype=bool)
        endType_full = np.zeros((N_dim, 3), dtype=bool)

        for i in tqdm(range(t_len-1)):
            idx = np.where(alive)[0]
            if len(idx) == 0: break
            t_i = t_arr[i]
            y_i = y_arr[i, :, idx]

            k1 = f(t_i, y_i)
            k2 = f(t_i + 0.5*h, y_i + 0.5*h*k1)
            k3 = f(t_i + 0.5*h, y_i + 0.5*h*k2)
            k4 = f(t_i + h, y_i + h*k3)

            y_i1 = y_i + h*(k1 + 2*k2 + 2*k3 + k4)/6
            y_arr[i+1, :, idx] = y_i1.T

            terminated, endType, disk_info_i = terminalCondition(y_i1, y_i, h=h)

            dead_idx = idx[terminated]
            endType_full[dead_idx] = endType[terminated]
            alive[dead_idx] = False

            y_arr[i+1, :, dead_idx] = y_arr[i, :, dead_idx]

            disk_mask = endType[terminated, 2]
            disk_idx = dead_idx[disk_mask]
            disk_info[disk_idx, :] = disk_info_i[terminated, :][disk_mask, :]

        return t_arr, y_arr, endType_full, disk_info
    
    # do not record the history of geodesics
    # for optimization
    else:
        y = initial.copy() # (y_dim, N_dim)
        disk_info = np.zeros((N_dim, 2))

        alive = np.ones(N_dim, dtype=bool)
        endType_full = np.zeros((N_dim, 3), dtype=bool)

        for t in tqdm(t_arr):
            idx = np.where(alive)[0]
            if len(idx) == 0: break
            y_i = y[:, idx].copy()

            k1 = f(t, y_i)
            k2 = f(t + 0.5*h, y_i + 0.5*h*k1)
            k3 = f(t + 0.5*h, y_i + 0.5*h*k2)
            k4 = f(t + h, y_i + h*k3)

            y_i1 = y_i + h*(k1 + 2*k2 + 2*k3 + k4)/6
            y[:, idx] = y_i1

            terminated, endType, disk_info_i = terminalCondition(y_i1, y_i, h=h)

            dead_idx = idx[terminated]
            endType_full[dead_idx] = endType[terminated]
            alive[dead_idx] = False

            disk_mask = endType[terminated, 2]
            disk_idx = dead_idx[disk_mask]
            disk_info[disk_idx, :] = disk_info_i[terminated, :][disk_mask, :]

        return None, None, endType_full, disk_info
