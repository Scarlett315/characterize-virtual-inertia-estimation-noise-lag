
import numpy as np
import pandas as pd
from statsmodels.tools.eval_measures import rmse

from liu_inertia_estimation import calc_inertia
from algebraic_inertia_estimation import calc_inertia_algebraic

DIST_TIME = 5

def calculate_metrics(ss, metrics_df, M_series, record_M=True):

    # ----- calculations -----

    tr = ss.REGF2.get("Tr", 1)

    # get needed timeseries
    freq_timeseries = ss.TDS.get_timeseries(ss.BusROCOF.f)
    rocof_timeseries = ss.TDS.get_timeseries(ss.BusROCOF.Wf_y) # df/dt
    power_timeseries = ss.TDS.get_timeseries(ss.REGF2.Pe)

    tds_df = pd.concat([freq_timeseries, rocof_timeseries, power_timeseries], axis=1)
    tds_df.columns = ["f", "df/dt", "p"]

    # initial metrics calculation
    peak_inertial_power = np.max(power_timeseries) 
    max_rocof = np.max(np.abs(rocof_timeseries))
    freq_nadir = np.min(freq_timeseries)
    p_avail = ss.REGF2.get("Pmax", 1) # pu
    p_ref = ss.PV.get("p0", 1) # pu

    # calculate inertia metrics
    M_t = calc_inertia_algebraic(tds_df, p_ref)
    #settling_time = calc_settling_time(ivp_res, "M")

    error = calculate_inertia_rmse(ss, M_t, DIST_TIME + 0.2)

    # add to metrics dataframe
    row = [tr, p_ref, peak_inertial_power, p_avail, error, max_rocof, freq_nadir]
    metrics_df.loc[len(metrics_df)] = row

    # add H timeseries (for a few select values of Tr)
    if record_M:
        M_series.loc[:,tr] = M_t

    return metrics_df, M_series



def calculate_metrics_ret(ss):   

    # ----- calculations -----

    tr = ss.REGF2.get("Tr", 1)

    # get needed timeseries
    freq_timeseries = ss.TDS.get_timeseries(ss.BusROCOF.f)
    rocof_timeseries = ss.TDS.get_timeseries(ss.BusROCOF.Wf_y) # df/dt
    power_timeseries = ss.TDS.get_timeseries(ss.REGF2.Pe)

    tds_df = pd.concat([freq_timeseries, rocof_timeseries, power_timeseries], axis=1)
    tds_df.columns = ["f", "df/dt", "p"]

    # initial metrics calculation
    peak_inertial_power = np.max(power_timeseries) 
    max_rocof = np.max(np.abs(rocof_timeseries))
    freq_nadir = np.min(freq_timeseries)
    p_avail = ss.REGF2.get("Pmax", 1) # pu
    p_ref = ss.PV.get("p0", 1) # pu

    # calculate inertia metrics
    M_t = calc_inertia_algebraic(tds_df, p_ref)
    #settling_time = calc_settling_time(ivp_res, "M")

    error = calculate_inertia_rmse(ss, M_t, DIST_TIME + 0.3)

    # add to metrics dataframe
    row = pd.DataFrame({
        "Tr (s)": [tr], 
        "Seed": [None], # fake :>
        "Pref": [p_ref], 
        "Peak Inertial Power": [peak_inertial_power], 
        "Power Available":[p_avail], 
        "H_RMSE (s)":[error], 
        "Max RoCoF":  [max_rocof], 
        "Nadir": [freq_nadir]
    })

    return row, M_t


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


def calculate_inertia_rmse(ss, M_ser, t_start):
    M_gfm = ss.REGF2.get("mf", 1)

    predicted = np.asarray(M_ser[t_start:])
    actual = np.full(predicted.size, M_gfm)
    error = rmse(actual, predicted)
    return error