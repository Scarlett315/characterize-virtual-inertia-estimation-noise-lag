def calc_inertia_algebraic(tds_df, p0, t0 = 5.1, tf=6):
    """
    Simple point-by-point H calculation.  
    tds_df should include cols "df/dt" and "p" with simulation time as the index.
    
    Returns time-series dataframe with column "H".
    """
    
    # H = 1/2 * (delta P)/(rocof) = 1/2 M 

    delta_p = tds_df.loc[:,"p"] - p0
    rocof = tds_df.loc[:,"df/dt"]

    h = -0.5 * delta_p.div(rocof)
    h = h.fillna(0)

    h_cut = h[t0:tf]

    return h_cut

