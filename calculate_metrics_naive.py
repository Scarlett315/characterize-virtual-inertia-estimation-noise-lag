# setting up ANDES

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.eval_measures import rmse
import andes


andes.config_logger(stream_level=30)

STEP_TIME = 0.01

def calculate_inertia_timeseries(ss, power_tms, freq_tms,p_ref, rocof_tms, window=20):
    deltaP = power_tms.subtract(p_ref)
    deltaF = freq_tms.subtract(1)

    tds_df = pd.concat([deltaP, deltaF, rocof_tms], axis=1)
    tds_df.columns = ["Delta P (pu)", "Delta F (pu)", "RoCoF (pu)"]

    time_invalid = 1 + window * STEP_TIME # will produce invalid results w/in the first window

    X = sm.add_constant(tds_df[['RoCoF (pu)', 'Delta F (pu)']])  # add Δf as 2nd predictor
    model = RollingOLS(tds_df['Delta P (pu)'], X, window=window)
    res = model.fit()

    H_t_raw = -res.params['RoCoF (pu)'] / 2 
    D_t_raw = res.params['Delta F (pu)']

    tds_df.loc[:,"H (s)"] = H_t_raw[time_invalid:]
    tds_df.loc[:,"D"] = D_t_raw[time_invalid:]

    return tds_df

def calculate_inertia_rmse(ss, tds_df, window=20):
    H_gfm = ss.REGF2.get("mf", 1)
    H_t = tds_df["H (s)"]
    time_invalid = 1 + window * STEP_TIME # will produce invalid results w/in the first window

    predicted = np.asarray(H_t[time_invalid:])
    actual = np.full(predicted.size, H_gfm)
    error = rmse(actual, predicted)
    return error

def calc_settling_time(df, var, threshold=0.05):
    y = np.asarray(df.loc[:,var])
    t = df.index

    steady_state = np.average(y[-10:-1]) # average last 10 vals as steady-state
    allowed_error = threshold * abs(steady_state)
    
    error = np.abs(y - steady_state)
    
    # find all indices where the error exceeds the threshold
    outside_band_indices = np.where(error > allowed_error)[0]
    
    if len(outside_band_indices) == 0:
        return t[0] 
        
    last_outside_index = outside_band_indices[-1]
    
    # ensure we don't overflow the array boundary
    if last_outside_index < len(t) - 1:
        return t[last_outside_index + 1]
    else:
        return t[last_outside_index]

def calculate_rocof(freq_timeseries):
    rocof_timeseries = pd.DataFrame(np.gradient(np.asarray(freq_timeseries), freq_timeseries.index)) # df/dt
    rocof_timeseries.index = freq_timeseries.index
    return rocof_timeseries

def calculate_metrics(ss, record_H=True):
    # ----- create empty dataframes -----
    metrics_df = pd.DataFrame({
    "Tr (s)": [], 
    "Pref": [],
    "Peak Inertial Power": [],
    "Power Available":[],
    "H_RMSE (s)": [],
    "Max RoCoF":[],
    "Nadir":[],
    "Settling Time (s)":[]
    })

    # save a few time-series for visualization
    h_series = pd.DataFrame({
    })

    # ----- calculations -----

    tr = ss.REGF2.get("Tr", 1)
    window = 20 # Baruzzi et al. used 1 window = 20 steps

    # get needed timeseries
    freq_timeseries = ss.TDS.get_timeseries(ss.BusFreq.f).loc[:,"BusFreq_1"]
    rocof_timeseries = calculate_rocof(freq_timeseries) # df/dt
    power_timeseries = ss.TDS.get_timeseries(ss.REGF2.Pe)

    # initial metrics calculation
    peak_inertial_power = np.max(power_timeseries) 
    max_rocof = np.max(np.abs(rocof_timeseries))
    freq_nadir = np.min(freq_timeseries)
    p_avail = ss.REGF2.get("Pmax", 1) # pu
    p_ref = ss.PV.get("p0", 1) # pu

    # calculate inertia metrics
    tds_df = calculate_inertia_timeseries(ss, power_timeseries, freq_timeseries, p_ref, rocof_timeseries, window=window)
    error = calculate_inertia_rmse(ss, tds_df, window=window)
    settling_time = calc_settling_time(tds_df, "H (s)")

    # add to metrics dataframe
    row = [tr, p_ref, peak_inertial_power, p_avail, error, max_rocof, freq_nadir, settling_time]
    metrics_df.loc[len(metrics_df)] = row

    # add H timeseries (for a few select values of Tr)
    if record_H:
        h_series.loc[:,tr] = tds_df.loc[:,"H (s)"]

    return metrics_df, h_series


