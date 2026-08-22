from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import andes

andes.config_logger(stream_level=30)

# ---- constants -------
TM = 0.104
TD = 10e-4

eps_x = 10e-5
eps_w = 10e-5

# deadband
def gamma(x, eps):
    return np.where(x >= eps, -1.0, np.where(x <= -eps, 1.0, 0.0))

# ---- calculate inertia ------
def calc_inertia(tds_df, tf):
    """
    Adapted from Liu et al. "On-Line Inertia Estimation for Synchronous and Non-Synchronous Devices", specifically their E2 method, which takes into account the controller's damping and primary frequency control. 
    This helps to remove oscillations in their estimated inertia. 

    tds_df should include cols "df/dt" and "p" with simulation time as the index.
    
    Returns time-series dataframe with each variable.
    """

    # solver function
    def rhs(t, y, TM, TD, eps_x, eps_w):
        Mstar, Dstar, domega_int, dp_int = y

        dp = dp_dt(t)
        d2w = d2omega_dt2(t)
        dw = domega_dt(t)

        dMstar = gamma(d2w, eps_x) * (dp - Mstar*d2w - Dstar*dw) / TM
        dDstar = gamma(domega_int, eps_w) * (dp_int - Mstar*dw - Dstar*domega_int) / TD
        ddomega_int = dw     
        ddp_int = dp  
    
        return [dMstar, dDstar, ddomega_int, ddp_int]

    # calculate RoCoF, derivative of RoCoF (RoRoCOF?? lol), and derivative of power (RoCoP)
    # need to interpolate between values b/c continuous-- called by solver

    domega_dt = CubicSpline(tds_df.index, tds_df.loc[:,"df/dt"])      
    d2omega_dt2 = domega_dt.derivative()               
    dp_dt = CubicSpline(tds_df.index, tds_df.loc[:,"p"]).derivative()

    # solve ivp
    sol = solve_ivp(rhs, [0, tf], y0=[0, 0, 0, 0],
                 args=(TM, TD, eps_x, eps_w),
                 method='Radau', dense_output=True,
                 t_eval=tds_df.index)

    # formatting
    vals_df = pd.DataFrame(sol.y[:2]).T
    vals_df.index = sol.t
    vals_df.columns = ["M", "D"]

    return vals_df

