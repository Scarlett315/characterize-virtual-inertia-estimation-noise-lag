import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
project_dir = Path.cwd()
test_case_dir = project_dir / "test_cases" # all test case excel sheets
disturbance_profile_dir = project_dir / "disturbance_profiles" # disturbance profiles for ANDES perturbation
out_dir = project_dir / "out"

rng = None

STEP_TIME = 0.001 # 1 ms
SIGMA_ROCOF, TAU_OU = 1e-4, 0.02 # deviations + time-step 

def ou_noise(n, dt=STEP_TIME, sigma=SIGMA_ROCOF, tau=TAU_OU, seed=0):
    """
    Exact simulation of Ornstein-Uhlenbeck Process (Gillepsie)
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-dt / tau)
    diffusion = sigma * np.sqrt(1.0 - decay ** 2)
    z = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = sigma * z[0]
    for i in range(1, n):
        x[i] = x[i - 1] * decay + diffusion * z[i]
    return x

"""
this was the original within the simulation, but now it's not needed
redone bcs it ran wayyy too slowly

# reset
def set_seed(seed):
    global rng
    rng = np.random.default_rng(seed)

# Orenstein-Uhlenbeck Simulation 
def ou_step(x, dt, tau, sigma):
    decay = np.exp(-dt / tau)
    return x * decay + sigma * np.sqrt(1 - decay**2) * rng.standard_normal()


# create a noise profile
def create_noise_profile(timestamps, seed, step_time=STEP_TIME, tau=TAU_OU, sigma=SIGMA_ROCOF):
    noise = pd.Series()
    ou_state = 0.0

    set_seed(seed)

    for i, _ in enumerate(timestamps):
        ou_state = ou_step(ou_state, step_time, tau, sigma)
        noise[i] = ou_state

    return noise
"""