# ------------------------------ Library ------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ Functions ------------------------------
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

        y_i1 = y_arr[i-1]
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
