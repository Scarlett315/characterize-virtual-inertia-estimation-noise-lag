import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- default parameters -----------------
KP_FILT_def, KI_FILT_def, TF_FILT_def = 50.0, 1.0, 1e-4   # Liu et al. Fig. 3, unmodified


EPS_O_def = 1e-6
T_M_def = 0.001    # paper's VSG-case default
EPS_OMEGADOT_def = 1e-6
T_M2_def, T_D2_def = 0.001, 1e-4   # paper's VSG-case defaults
EPS_DOMEGA_def = 1e-6

# ---------------- Utils (PI Filter & gamma!) ------------------

def pi_filter(u, dt_arr, Kp=KP_FILT_def, Ki=KI_FILT_def, Tf=TF_FILT_def):
    """PI pre-filter: smooths RoCoF (u) into a filtered RoCoF (x2) and a
    filtered RoCoCoF (x1), replacing naive double np.gradient."""
    x1, x2 = np.zeros(len(u)), np.zeros(len(u))
    for i in range(1, len(u)):
        dt = dt_arr[i]
        denom = 1 + dt / Tf + dt ** 2 * Kp * Ki / Tf
        x1[i] = (x1[i - 1] + dt * (Kp / Tf) * (u[i] - x2[i - 1])) / denom
        x2[i] = x2[i - 1] + dt * Ki * x1[i]
    return x2, x1


def run_pi_filter(t_est, omega_fdf, KP_FILT=KP_FILT_def, KI_FILT=KI_FILT_def, TF_FILT=TF_FILT_def):
    dt_arr = np.diff(t_est, prepend=t_est[0])
    domega_dt_raw = np.gradient(omega_fdf, t_est)

    domega_dt, d2omega_dt2 = pi_filter(domega_dt_raw, dt_arr, KP_FILT, KI_FILT, TF_FILT)
    return domega_dt, d2omega_dt2

def lowpass_filter(u, dt_arr, tau):
    y = np.zeros(len(u))
    for i in range(1, len(u)):
        a = dt_arr[i] / tau
        y[i] = (y[i-1] + a * u[i]) / (1 + a)

    return y, (u - y) / tau



def gamma(x, eps_x):
    return np.where(x >= eps_x, -1.0, np.where(x <= -eps_x, 1.0, 0.0))

# ---------------- E0, E1, and E2 estimators -------------
def run_E0(dp_dt, omega_ddot, eps_o=EPS_OMEGADOT_def):
    """Direct division. Holds the previous value when the denominator
    is too small to divide by."""
    M = np.zeros(len(dp_dt))
    for i in range(len(dp_dt)):
        if abs(omega_ddot[i]) >= eps_o:
            M[i] = -dp_dt[i] / omega_ddot[i]
        else:
            M[i] = M[i - 1] if i > 0 else 0.0
    return M


def run_E1(dp_dt, omega_ddot, t, T_M=T_M_def, eps=EPS_OMEGADOT_def):
    """Gated integration, backward Euler."""
    M = np.zeros(len(t))
    for i in range(1, len(t)):
        dt_i = t[i] - t[i - 1]
        g = gamma(omega_ddot[i], eps)
        a = g * dp_dt[i] / T_M
        b = g * omega_ddot[i] / T_M
        M[i] = (M[i - 1] + dt_i * a) / (1 - dt_i * b)
    return M


def run_E2(dp_dt, omega_dot, omega_ddot, delta_omega, delta_p, t,
           T_M=T_M2_def, T_D=T_D2_def, eps=EPS_OMEGADOT_def):
    """Coupled 2x2 solve for M and D. Only M is used by this study."""
    M, D = np.zeros(len(t)), np.zeros(len(t))
    for i in range(1, len(t)):
        dt_i = t[i] - t[i - 1]
        gM, gD = gamma(omega_ddot[i], eps), gamma(delta_omega[i], eps)
        A, B, C = gM * dp_dt[i] / T_M, gM * omega_ddot[i] / T_M, gM * omega_dot[i] / T_M
        E, F, G = gD * delta_p[i] / T_D, gD * omega_dot[i] / T_D, gD * delta_omega[i] / T_D
        m11, m12, m21, m22 = 1 - dt_i * B, -dt_i * C, -dt_i * F, 1 - dt_i * G
        r1, r2 = M[i - 1] + dt_i * A, D[i - 1] + dt_i * E
        det = m11 * m22 - m12 * m21
        M[i] = (r1 * m22 - m12 * r2) / det
        D[i] = (m11 * r2 - r1 * m21) / det
    return M, D

# --------------- run all 3! ----------------
def run_all_estimators(omega, p, t, tau, noise=None):
    dt_arr = np.diff(t, prepend=t[0])
    dp_dt = np.gradient(p, t)

    rocof_raw = np.gradient(omega, t)
    # addition of noise to RoCoF signal
    if noise is not None:
        rocof_raw += noise

    # lowpass filter
    rocof, rocof_dot = lowpass_filter(rocof_raw, dt_arr, tau)

    # delta P
    delta_p = p - p[0]

    # run all 3 estimators 
    delta_omega = np.concatenate([[0.0], np.cumsum((rocof[1:] + rocof[:-1]) / 2 * np.diff(t))])
    
    M_E0 = run_E0(dp_dt, rocof_dot)
    M_E1 = run_E1(dp_dt, rocof_dot, t)
    M_E2, _ = run_E2(dp_dt, rocof, rocof_dot, delta_omega, delta_p, t)

    return {'E0': M_E0 / 2, 'E1': M_E1 / 2, 'E2': M_E2 / 2}

# ------------------- Visualizations ------------------
def vis_est_inertia(t_est, H_est, H_true, title, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, H_est, '.', markersize=2, color='tab:blue')
    ax.axhline(H_true, color='k', linewidth=1.2, linestyle='--', label=f'H_true = {H_true} s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Estimated H (s)')
    ax.set_title(title)
    ax.legend(loc='best')

    if xlim != None:
        ax.set_xlim(xlim)

    if ylim != None:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()

    return True