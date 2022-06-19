import pandas as pd
import numpy as np
from delta_figure import to_year_fraction, calc_precip, calc_q, plot_del_figure
from preprocessing import convert_datetime, list_years, find_start_sampling, list_unusable_dates, \
    split_fluxes_by_hydro_year, split_isotopes_by_hydro_year
from cleaning import sum_precipitation_and_runoff, remove_nan_samples
from calculations import endsplit
from plot import calculate_wls, undercatch_correction, calculate_fractions, plot_panels, plot_et_amounts, calculate_avg_et, calculate_scaled_et
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHB.csv')
daily_data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHBDaily.csv')

# Isotope data variables
sampling_dates = data.loc[:, 'date']
precip_isotope = data.loc[:, 'Precip_18O_combined']
interval = data.loc[:, 'time_interval_d']

# Flux data variables
date_daily = daily_data['date'].to_list()
p = daily_data['Sum(Precip_mm)'].tolist()
temperature = daily_data['Mean(airtemp_C)'].tolist()

wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month = \
    calc_precip(sampling_dates, p, precip_isotope, interval, 4, 9)

def workflow_endsplit(dates, precip_isotope, sampling_interval, stream_isotope, dates_daily, p, q, temperature):
    sampling_dates = convert_datetime(dates)
    date_daily = convert_datetime(dates_daily)
    years = list_years(sampling_dates)
    years_daily = list_years(date_daily)
    first_month, first_year = find_start_sampling(sampling_dates, date_daily, precip_isotope, p, stream_isotope, q,
                                       side='start')

    last_month, last_year = find_start_sampling(sampling_dates[::-1], date_daily[::-1], precip_isotope.values[::-1],
                                       p[::-1], stream_isotope.values[::-1], q[::-1], side='end')

    no_count_date = list_unusable_dates(sampling_dates, first_month, years.index(int(first_year)), last_month,
                                      years.index(int(last_year)), years)

    no_count_date_daily = list_unusable_dates(date_daily, first_month, years_daily.index(int(first_year)), last_month,
                                      years_daily.index(int(last_year)), years_daily)

    fluxes = split_fluxes_by_hydro_year(date_daily, years_daily,no_count_date_daily, p, q, temperature)

    date_by_year, precip_d_year, pdelcat_with_nan, runoff_d_year, qdelcat_with_nan, interval_by_year = \
                                     split_isotopes_by_hydro_year(sampling_dates, sampling_interval, years,
                                     no_count_date, precip_isotope, stream_isotope, first_year)

    interval = [[] for _ in fluxes]
    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            interval[i].append(1)

    pwt, qwt = sum_precipitation_and_runoff(date_by_year, fluxes, precip_d_year, runoff_d_year)

    qdel, qdelcat = remove_nan_samples(years, runoff_d_year, qdelcat_with_nan)

    pdel, pdelcat = remove_nan_samples(years, precip_d_year, qdelcat_with_nan)

    # Analysis using values from year of interest (assumes no lag or mixing)
    columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se', 'ET', 'ET_se',
             'AllP_del', 'f_Ps', 'f_Ps_se']
    no_lag = []
    qwt_list = []
    qdel_list = []
    qdate_list = []
    for i in range(len(years)):
        if pwt[i]:
            for s in range(len(qwt[i])):
                qwt_list.append(qwt[i][s]/10)
                qdel_list.append(qdel[i][s])
                qdate_list.append(date_by_year[i][s])
            new_row = endsplit(pdel[i], qdel[i], pwt[i], qwt[i], pdelcat[i], qdelcat[i], fluxes[i]['P'],
                                       fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            no_lag.append(new_row)
    df_no_lag = pd.DataFrame(data=no_lag, columns=columns)
    no_lag_ci_f_et = calculate_wls(df_no_lag['P_s'], df_no_lag['f_ET'], df_no_lag['f_ET_se'])
    no_lag_ci_f_ps = calculate_wls(df_no_lag['P_s'], df_no_lag['f_Ps'], df_no_lag['f_Ps_se'])
    no_lag_ci = {'P_s': no_lag_ci_f_et['P_s'],
                 'f_ET': no_lag_ci_f_et['f'],
                 'f_Ps': no_lag_ci_f_ps['f'],
                 'x_val_ci': no_lag_ci_f_et['x_val_ci'],
                 'f_ET_fitvals': no_lag_ci_f_et['fitvals'],
                 'f_ET_ci_low': no_lag_ci_f_et['ci_low'],
                 'f_ET_ci_upp': no_lag_ci_f_et['ci_upp'],
                 'f_Ps_fitvals': no_lag_ci_f_ps['fitvals'],
                 'f_Ps_ci_low': no_lag_ci_f_ps['ci_low'],
                 'f_Ps_ci_upp': no_lag_ci_f_ps['ci_upp']}

    # Lagged flow analysis, 1 year
    # End-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest
    lag_1 = []
    LAG_YRS = 1
    for i in range(LAG_YRS, len(years)):
        if pwt[i-LAG_YRS] and fluxes[i]['Pcat']:
            new_row = endsplit(pdel[i-LAG_YRS], qdel[i], pwt[i-LAG_YRS], qwt[i], pdelcat[i-LAG_YRS], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            lag_1.append(new_row)
    df_lag_1 = pd.DataFrame(data=lag_1, columns=columns)
    lag_1_ci_f_et = calculate_wls(df_lag_1['P_s'], df_lag_1['f_ET'], df_lag_1['f_ET_se'])
    lag_1_ci_f_ps = calculate_wls(df_no_lag['P_s'], df_lag_1['f_Ps'], df_lag_1['f_Ps_se'])
    lag_1_ci = {'P_s': lag_1_ci_f_et['P_s'],
                'f_ET': lag_1_ci_f_et['f'],
                'f_Ps': lag_1_ci_f_ps['f'],
                'x_val_ci': lag_1_ci_f_et['x_val_ci'],
                'f_ET_fitvals': lag_1_ci_f_et['fitvals'],
                'f_ET_ci_low': lag_1_ci_f_et['ci_low'],
                'f_ET_ci_upp': lag_1_ci_f_et['ci_upp'],
                'f_Ps_fitvals': lag_1_ci_f_ps['fitvals'],
                'f_Ps_ci_low': lag_1_ci_f_ps['ci_low'],
                'f_Ps_ci_upp': lag_1_ci_f_ps['ci_upp']}

    # Lagged flow analysis, 2 years
    # End-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest
    lag_2 = []
    LAG_YRS = 2
    for i in range(LAG_YRS, len(years)):
        if pwt[i-LAG_YRS] and fluxes[i]['Pcat']:
            new_row = endsplit(pdel[i - LAG_YRS], qdel[i], pwt[i - LAG_YRS], qwt[i], pdelcat[i - LAG_YRS],
                               qdelcat[i], fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            lag_2.append(new_row)
    df_lag_2 = pd.DataFrame(data=lag_2, columns=columns)
    lag_2_ci_f_et = calculate_wls(df_lag_2['P_s'], df_lag_2['f_ET'], df_lag_2['f_ET_se'])
    lag_2_ci_f_ps = calculate_wls(df_lag_2['P_s'], df_lag_2['f_Ps'], df_lag_2['f_Ps_se'])
    lag_2_ci = {'P_s': lag_2_ci_f_et['P_s'],
                'f_ET': lag_2_ci_f_et['f'],
                'f_Ps': lag_2_ci_f_ps['f'],
                'x_val_ci': lag_2_ci_f_et['x_val_ci'],
                'f_ET_fitvals': lag_2_ci_f_et['fitvals'],
                'f_ET_ci_low': lag_2_ci_f_et['ci_low'],
                'f_ET_ci_upp': lag_2_ci_f_et['ci_upp'],
                'f_Ps_fitvals': lag_2_ci_f_ps['fitvals'],
                'f_Ps_ci_low': lag_2_ci_f_ps['ci_low'],
                'f_Ps_ci_upp': lag_2_ci_f_ps['ci_upp']}

    # Mixed groundwater analysis
    # Repeat end-member splitting, but keep constant values of pdel_bar each season for all years using the summer
    # and winter isotope values averaged over the entire study period
    isotope_vals = []
    weights = []
    season = []
    for i in range(len(pwt)):
        for d in range(len(pwt[i])):
            isotope_vals.append(pdel[i][d])
            weights.append(pwt[i][d])
            season.append(pdelcat[i][d])
    mixed = []
    for i in range(len(years)):
        if pwt[i]:
            new_row = endsplit(isotope_vals, qdel[i], weights, qwt[i], season, qdelcat[i], fluxes[i]['P'],
                        fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            mixed.append(new_row)
    df_mixed = pd.DataFrame(data=mixed, columns=columns)
    mixed_ci_f_et = calculate_wls(df_mixed['P_s'], df_mixed['f_ET'], df_mixed['f_ET_se'])
    mixed_ci_f_ps = calculate_wls(df_mixed['P_s'], df_mixed['f_Ps'], df_mixed['f_Ps_se'])
    mixed_ci = {'P_s': mixed_ci_f_et['P_s'],
                'f_ET': mixed_ci_f_et['f'],
                'f_Ps': mixed_ci_f_ps['f'],
                'x_val_ci': mixed_ci_f_et['x_val_ci'],
                'f_ET_fitvals': mixed_ci_f_et['fitvals'],
                'f_ET_ci_low': mixed_ci_f_et['ci_low'],
                'f_ET_ci_upp': mixed_ci_f_et['ci_upp'],
                'f_Ps_fitvals': mixed_ci_f_ps['fitvals'],
                'f_Ps_ci_low': mixed_ci_f_ps['ci_low'],
                'f_Ps_ci_upp': mixed_ci_f_ps['ci_upp']}

    # Analysis using precipitation delta values averaged over current and previous year
    MERGE_YRS = 1
    merge_pdel_1 = [[] for _ in pwt]
    merge_weights_1 = [[] for _ in pwt]
    merge_season_1 = [[] for _ in pwt]
    for i in range(MERGE_YRS, len(pwt)):
        for d in range(len(pwt[i])):
            merge_pdel_1[i].append(pdel[i][d])
            merge_weights_1[i].append(pwt[i][d])
            merge_season_1[i].append(pdelcat[i][d])
        for d in range(len(pwt[i-1])):
            merge_pdel_1[i].append(pdel[i-1][d])
            merge_weights_1[i].append(pwt[i-1][d])
            merge_season_1[i].append(pdelcat[i-1][d])
    lag_1_mean = []
    for i in range(MERGE_YRS, len(years)):
        if pwt[i]:
            new_row = endsplit(merge_pdel_1[i], qdel[i], merge_weights_1[i], qwt[i], merge_season_1[i], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            lag_1_mean.append(new_row)
    df_lag_1_mean = pd.DataFrame(data=lag_1_mean, columns=columns)
    lag_1_mean_ci_f_et = calculate_wls(df_lag_1_mean['P_s'], df_lag_1_mean['f_ET'], df_lag_1_mean['f_ET_se'])
    lag_1_mean_ci_f_ps = calculate_wls(df_lag_1_mean['P_s'], df_lag_1_mean['f_Ps'], df_lag_1_mean['f_Ps_se'])
    lag_1_mean_ci = {'P_s': lag_1_mean_ci_f_et['P_s'],
                     'f_ET': lag_1_mean_ci_f_et['f'],
                     'f_Ps': lag_1_mean_ci_f_ps['f'],
                     'x_val_ci': lag_1_mean_ci_f_et['x_val_ci'],
                     'f_ET_fitvals': lag_1_mean_ci_f_et['fitvals'],
                     'f_ET_ci_low': lag_1_mean_ci_f_et['ci_low'],
                     'f_ET_ci_upp': lag_1_mean_ci_f_et['ci_upp'],
                     'f_Ps_fitvals': lag_1_mean_ci_f_ps['fitvals'],
                     'f_Ps_ci_low': lag_1_mean_ci_f_ps['ci_low'],
                     'f_Ps_ci_upp': lag_1_mean_ci_f_ps['ci_upp']}

    # Analysis using precipitation delta values averaged over current and previous 2 years
    merge_years = 2
    merge_pdel_2 = [[] for _ in pwt]
    merge_weights_2 = [[] for _ in pwt]
    merge_season_2 = [[] for _ in pwt]
    for i in range(merge_years, len(pwt)):
        for d in range(len(pwt[i])):
            merge_pdel_2[i].append(pdel[i][d])
            merge_weights_2[i].append(pwt[i][d])
            merge_season_2[i].append(pdelcat[i][d])
        for d in range(len(pwt[i - 1])):
            merge_pdel_2[i].append(pdel[i - 1][d])
            merge_weights_2[i].append(pwt[i - 1][d])
            merge_season_2[i].append(pdelcat[i - 1][d])
        for d in range(len(pwt[i-2])):
            merge_pdel_2[i].append(pdel[i - 2][d])
            merge_weights_2[i].append(pwt[i - 2][d])
            merge_season_2[i].append(pdelcat[i - 2][d])
    lag_2_mean = []
    for i in range(merge_years, len(years)):
        if pwt[i]:
            new_row = endsplit(merge_pdel_1[i], qdel[i], merge_weights_1[i], qwt[i], merge_season_1[i], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            new_row[0] = years[i]
            lag_2_mean.append(new_row)
    df_lag_2_mean = pd.DataFrame(data=lag_2_mean, columns=columns)
    lag_2_mean_ci_f_et = calculate_wls(df_lag_2_mean['P_s'], df_lag_2_mean['f_ET'], df_lag_2_mean['f_ET_se'])
    lag_2_mean_ci_f_ps = calculate_wls(df_lag_2_mean['P_s'], df_lag_2_mean['f_Ps'], df_lag_2_mean['f_Ps_se'])
    lag_2_mean_ci = {'P_s': lag_2_mean_ci_f_et['P_s'],
                     'f_ET': lag_2_mean_ci_f_et['f'],
                     'f_Ps': lag_2_mean_ci_f_ps['f'],
                     'x_val_ci': lag_2_mean_ci_f_et['x_val_ci'],
                     'f_ET_fitvals': lag_2_mean_ci_f_et['fitvals'],
                     'f_ET_ci_low': lag_2_mean_ci_f_et['ci_low'],
                     'f_ET_ci_upp': lag_2_mean_ci_f_et['ci_upp'],
                     'f_Ps_fitvals': lag_2_mean_ci_f_ps['fitvals'],
                     'f_Ps_ci_low': lag_2_mean_ci_f_ps['ci_low'],
                     'f_Ps_ci_upp': lag_2_mean_ci_f_ps['ci_upp']}
    df = {"No Lag": df_no_lag, "Lag 1": df_lag_1, "Lag 2": df_lag_2, "Mixed": df_mixed, "Lag 1 Mean":df_lag_1_mean,
            "Lag 2 Mean": df_lag_2_mean}
    df_ci = {"No Lag":no_lag_ci, "Lag 1": lag_1_ci, "Lag 2": lag_2_ci, "Mixed": mixed_ci,
            "Lag 1 Mean": lag_1_mean_ci, "Lag 2 Mean": lag_2_mean_ci}

    return df, df_ci, qwt_list, qdel_list, qdate_list

# All Rietholzbach Catchment

stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
q = daily_data['Summed_RHB_discharge_mm'].tolist()

df_all, ci_all_method_1, qwt_all, qdel_all, qdate_all = workflow_endsplit(sampling_dates, precip_isotope, interval,
            stream_isotope, date_daily, p, q, temperature)
wtd_mean_stream_all, error_stream_all = calc_q(q, stream_isotope, sampling_dates, date_daily)

plt.scatter(df_all["No Lag"]['Ptot'], df_all["No Lag"]['AllP_del'])
plt.show()

# Upper Rietholzbach

stream_isotope_upper = data.loc[:, 'Upper_RHB_18O_combined']
q_upper = daily_data['SummedDischarge(upper_RHB_mm/h)'].tolist()

df_upper, ci_upper_method_1, qwt_upper, qdel_upper, qdate_upper = workflow_endsplit(sampling_dates, precip_isotope,
            interval, stream_isotope_upper, date_daily, p, q_upper, temperature)
wtd_mean_stream_upper, error_stream_upper = calc_q(q_upper, stream_isotope_upper, sampling_dates, date_daily)

plt.scatter(df_upper["No Lag"]['Ptot'], df_upper["No Lag"]['AllP_del'])
plt.show()

# Lysimeter

Isotope_lysimeter_seepage = data.loc[:, 'Lysimeter_18O_combined']
Amount_lysimeter_seepage = daily_data['lys_seep_mm/day'].tolist()

df_lys, ci_lys_method_1, qwt_lys, qdel_lys, qdate_lys = workflow_endsplit(sampling_dates, precip_isotope, interval,
            Isotope_lysimeter_seepage, date_daily, p, Amount_lysimeter_seepage, temperature)
wtd_mean_stream_lys, error_stream_lys = calc_q(Amount_lysimeter_seepage, Isotope_lysimeter_seepage, sampling_dates,
                                              date_daily)

plt.scatter(df_lys["No Lag"]['Ptot'], df_lys["No Lag"]['AllP_del'])
plt.show()

# Figure showing delta values of precipitation and streamflow
wtd_mean_stream = [wtd_mean_stream_all, wtd_mean_stream_upper, wtd_mean_stream_lys]
error_stream = [error_stream_all, error_stream_upper, error_stream_lys]
stream_label = ["All", np.nan, "Upper and Lysimeter"] # Upper and Lysimeter average delta values overlap on plot
colors = ["blue", "orange", "green"]

date_all = []
for i in range(len(qdate_all)):
    date_all.append(to_year_fraction(qdate_all[i])*11)
date_upper = []
for i in range(len(qdate_upper)):
    date_upper.append(to_year_fraction(qdate_upper[i])*11)
date_lys = []
for i in range(len(qdate_lys)):
    date_lys.append(to_year_fraction(qdate_lys[i])*11)

plot_del_figure(4, 9, wtd_mean_summer, s_error_summer, wtd_mean_winter, s_error_winter, wtd_mean_per_month,
        s_error_per_month, wtd_mean_stream, stream_label, colors, date_all, date_upper, date_lys, qdel_all, qdel_upper,
        qdel_lys, qwt_all, qwt_upper, qwt_lys)

# Weighted Regression of Significant Slopes
x = [1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
     2012, 2013]
plot_panels(ci_all_method_1, ci_upper_method_1, ci_lys_method_1,
           'Calculated Using Measured Precipitation, Streamflow, and ET')

# Figure showing evapotranspiration from lysimeter weights
evapotranspiration = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\Evapotranspiration.csv')

# Figure showing annual ET amount by catchment and year
data = plot_et_amounts(x, df_all['No Lag'], df_upper['No Lag'], df_lys['No Lag'], evapotranspiration)

undercatch = {'Year':[], 'Undercatch (mm)':[]}
for i in range(len(x)):
    if data['Lysimeter discharge'][i] != 0 and data['Lysimeter weights'][i] != 0 and not np.isnan(
            data['Lysimeter discharge'][i]) and not np.isnan(data['Lysimeter weights'][i]):
        undercatch['Year'].append(x[i])
        undercatch['Undercatch (mm)'].append(data['Lysimeter weights'][i] - data['Lysimeter discharge'][i])
df_undercatch = pd.DataFrame(undercatch)

df_all_undercatch_adj = {"No Lag":undercatch_correction(df_all['No Lag'], df_undercatch),
                        "Lag 1": undercatch_correction(df_all['Lag 1'], df_undercatch),
                        "Lag 2": undercatch_correction(df_all['Lag 2'], df_undercatch),
                        "Mixed": undercatch_correction(df_all['Mixed'], df_undercatch),
                        "Lag 1 Mean": undercatch_correction(df_all['Lag 1 Mean'], df_undercatch),
                        "Lag 2 Mean":undercatch_correction(df_all['Lag 2 Mean'], df_undercatch)}

df_upper_undercatch_adj = {"No Lag": undercatch_correction(df_upper['No Lag'], df_undercatch),
                        "Lag 1": undercatch_correction(df_upper['Lag 1'], df_undercatch),
                        "Lag 2": undercatch_correction(df_upper['Lag 2'], df_undercatch),
                        "Mixed":undercatch_correction(df_upper['Mixed'], df_undercatch),
                        "Lag 1 Mean":undercatch_correction(df_upper['Lag 1 Mean'], df_undercatch),
                        "Lag 2 Mean":undercatch_correction(df_upper['Lag 2 Mean'], df_undercatch)}

df_lys_undercatch_adj = {"No Lag": undercatch_correction(df_lys['No Lag'], df_undercatch),
                        "Lag 1": undercatch_correction(df_lys['Lag 1'], df_undercatch),
                        "Lag 2": undercatch_correction(df_lys['Lag 2'], df_undercatch),
                        "Mixed":undercatch_correction(df_lys['Mixed'], df_undercatch),
                        "Lag 1 Mean":undercatch_correction(df_lys['Lag 1 Mean'], df_undercatch),
                        "Lag 2 Mean":undercatch_correction(df_lys['Lag 2 Mean'], df_undercatch)}

ci_all_method_2 = {"No Lag":calculate_fractions(df_all_undercatch_adj['No Lag'], et='mass bal'),
                   "Lag 1": calculate_fractions(df_all_undercatch_adj['Lag 1'], et='mass bal'),
                   "Lag 2": calculate_fractions(df_all_undercatch_adj['Lag 2'], et='mass bal'),
                   "Mixed": calculate_fractions(df_all_undercatch_adj['Mixed'], et='mass bal'),
                   "Lag 1 Mean": calculate_fractions(df_all_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                   "Lag 2 Mean": calculate_fractions(df_all_undercatch_adj['Lag 2 Mean'], et='mass bal')}

ci_upper_method_2 = {"No Lag": calculate_fractions(df_upper_undercatch_adj['No Lag'], et='mass bal'),
                    "Lag 1": calculate_fractions(df_upper_undercatch_adj['Lag 1'], et='mass bal'),
                    "Lag 2": calculate_fractions(df_upper_undercatch_adj['Lag 2'], et='mass bal'),
                    "Mixed": calculate_fractions(df_upper_undercatch_adj['Mixed'], et='mass bal'),
                    "Lag 1 Mean": calculate_fractions(df_upper_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                    "Lag 2 Mean": calculate_fractions(df_upper_undercatch_adj['Lag 2 Mean'], et='mass bal')}

ci_lys_method_2 = {"No Lag": calculate_fractions(df_lys_undercatch_adj['No Lag'], et='mass bal'),
                   "Lag 1": calculate_fractions(df_lys_undercatch_adj['Lag 1'], et='mass bal'),
                   "Lag 2": calculate_fractions(df_lys_undercatch_adj['Lag 2'], et='mass bal'),
                   "Mixed": calculate_fractions(df_lys_undercatch_adj['Mixed'], et='mass bal'),
                   "Lag 1 Mean": calculate_fractions(df_lys_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                   "Lag 2 Mean": calculate_fractions(df_lys_undercatch_adj['Lag 2 Mean'], et='mass bal')}

plot_panels(ci_all_method_2, ci_upper_method_2, ci_lys_method_2,
           'Calculated Using Undercatch_Adjusted Precipitation, Annually Mass-Balanced ET, and Measured Streamflow')

ci_all_method_3 = {"No Lag":calculate_fractions(calculate_scaled_et(df_all['No Lag'], evapotranspiration)),
                   "Lag 1": calculate_fractions(calculate_scaled_et(df_all['Lag 1'], evapotranspiration)),
                   "Lag 2": calculate_fractions(calculate_scaled_et(df_all['Lag 2'], evapotranspiration)),
                   "Mixed": calculate_fractions(calculate_scaled_et(df_all['Mixed'], evapotranspiration)),
                   "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_all['Lag 1 Mean'], evapotranspiration)),
                   "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_all['Lag 2 Mean'], evapotranspiration))}

ci_upper_method_3 = {"No Lag": calculate_fractions(calculate_scaled_et(df_upper['No Lag'], evapotranspiration)),
                     "Lag 1": calculate_fractions(calculate_scaled_et(df_upper['Lag 1'], evapotranspiration)),
                     "Lag 2": calculate_fractions(calculate_scaled_et(df_upper['Lag 2'], evapotranspiration)),
                     "Mixed": calculate_fractions(calculate_scaled_et(df_upper['Mixed'], evapotranspiration)),
                     "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_upper['Lag 1 Mean'], evapotranspiration)),
                     "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_upper['Lag 2 Mean'], evapotranspiration))}

ci_lys_method_3 = {"No Lag": calculate_fractions(calculate_scaled_et(df_lys['No Lag'], evapotranspiration)),
                   "Lag 1": calculate_fractions(calculate_scaled_et(df_lys['Lag 1'], evapotranspiration)),
                   "Lag 2": calculate_fractions(calculate_scaled_et(df_lys['Lag 2'], evapotranspiration)),
                   "Mixed": calculate_fractions(calculate_scaled_et(df_lys['Mixed'], evapotranspiration)),
                   "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_lys['Lag 1 Mean'], evapotranspiration)),
                   "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_lys['Lag 2 Mean'], evapotranspiration))}

plot_panels(ci_all_method_3, ci_upper_method_3, ci_lys_method_3,
           'Calculated Using Measured Precipitation, Lysimeter-Scaled ET, and Annually Mass-Balanced Streamflow')

ci_all_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 2 Mean'], evapotranspiration))}

ci_upper_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean":calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 2 Mean'], evapotranspiration))}

ci_lys_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 2 Mean'], evapotranspiration))}

plot_panels(ci_all_method_4, ci_upper_method_4, ci_lys_method_4,
           'Calculated Using Undercatch-Adjusted Precipitation, Lysimeter-Scaled ET, and Annually Mass-Balanced Streamflow')

ci_all_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_all['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_all['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_all['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_all['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_all['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_all['Lag 2 Mean']))}

ci_upper_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_upper['No Lag'])),
                     "Lag 1": calculate_fractions(calculate_avg_et(df_upper['Lag 1'])),
                     "Lag 2": calculate_fractions(calculate_avg_et(df_upper['Lag 2'])),
                     "Mixed": calculate_fractions(calculate_avg_et(df_upper['Mixed'])),
                     "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_upper['Lag 1 Mean'])),
                     "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_upper['Lag 2 Mean']))}

ci_lys_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_lys['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_lys['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_lys['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_lys['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_lys['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_lys['Lag 2 Mean']))}

plot_panels(ci_all_method_5, ci_upper_method_5, ci_lys_method_5,
           'Calculated Using Measured Precipitation, Average ET, and Annually Mass-Balanced Streamflow')

ci_all_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 2 Mean']))}

ci_upper_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['No Lag'])),
                     "Lag 1": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 1'])),
                     "Lag 2": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 2'])),
                     "Mixed": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Mixed'])),
                     "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 1 Mean'])),
                     "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 2 Mean']))}

ci_lys_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 2 Mean']))}

plot_panels(ci_all_method_6, ci_upper_method_6, ci_lys_method_6,
           'Calculated Using Undercatch-Adjusted Precipitation, Average ET, and Annually Mass-Balanced Streamflow')