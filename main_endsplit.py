import pandas as pd
import numpy as np
from delta_figure import to_year_fraction, calc_precip, calc_q, plot_del_figure
from preprocessing import convert_datetime, list_years, find_range_sampling, list_unusable_dates, \
    split_fluxes_by_hydro_year, split_isotopes_by_hydro_year
from cleaning import sum_precipitation_and_runoff, remove_nan_samples
from calculations import endsplit
from plot import calculate_wls, undercatch_correction, calculate_fractions, plot_panels, plot_et_amounts, \
    calculate_avg_et, calculate_scaled_et
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
import math
import copy
from textwrap import wrap


def workflow_endsplit(dates, precip_isotope, sampling_interval, stream_isotope, dates_daily, p, q, temperature, label):
    sampling_dates = convert_datetime(dates)
    date_daily = convert_datetime(dates_daily)
    years = list_years(sampling_dates)
    years_daily = list_years(date_daily)
    first_month, first_year = find_range_sampling(sampling_dates, date_daily, precip_isotope, p, stream_isotope, q,
                                       side='start')

    last_month, last_year = find_range_sampling(sampling_dates[::-1], date_daily[::-1], precip_isotope.values[::-1],
                                       p[::-1], stream_isotope.values[::-1], q[::-1], side='end')

    no_count_date = list_unusable_dates(sampling_dates, first_month, years.index(int(first_year)), last_month,
                                      years.index(int(last_year)), years)

    no_count_date_daily = list_unusable_dates(date_daily, first_month, years_daily.index(int(first_year)), last_month,
                                      years_daily.index(int(last_year)), years_daily)

    fluxes = split_fluxes_by_hydro_year(date_daily, years_daily, no_count_date_daily, p, q, temperature)

    date_by_year, precip_d_year, pdelcat_with_nan, runoff_d_year, qdelcat_with_nan, interval_by_year = \
                                     split_isotopes_by_hydro_year(sampling_dates, sampling_interval, years,
                                     no_count_date, precip_isotope, stream_isotope, first_year)

    pwt, qwt = sum_precipitation_and_runoff(date_by_year, fluxes, precip_d_year, runoff_d_year)
    qdel, qdelcat = remove_nan_samples(years, runoff_d_year, qdelcat_with_nan)
    pdel, pdelcat = remove_nan_samples(years, precip_d_year, qdelcat_with_nan)

    # Analysis using values from year of interest (assumes no lag or mixing)
    columns = ['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se', 'ET', 'ET_se',
             'AllP_del', 'f_Ps', 'f_Ps_se', 'P_w', 'P_w_se', 'ratio', 'ratio_se']

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
                                       fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            no_lag.append(new_row)
    df_no_lag = pd.DataFrame(data=no_lag, columns=columns)

    # Lagged flow analysis, 1 year
    # End-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest
    lag_1 = []
    LAG_YRS = 1
    for i in range(LAG_YRS, len(years)):
        if pwt[i-LAG_YRS] and fluxes[i]['Pcat']:
            new_row = endsplit(pdel[i-LAG_YRS], qdel[i], pwt[i-LAG_YRS], qwt[i], pdelcat[i-LAG_YRS], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            lag_1.append(new_row)
    df_lag_1 = pd.DataFrame(data=lag_1, columns=columns)

    # Lagged flow analysis, 2 years
    # End-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest
    lag_2 = []
    LAG_YRS = 2
    for i in range(LAG_YRS, len(years)):
        if pwt[i-LAG_YRS] and fluxes[i]['Pcat']:
            new_row = endsplit(pdel[i - LAG_YRS], qdel[i], pwt[i - LAG_YRS], qwt[i], pdelcat[i - LAG_YRS],
                               qdelcat[i], fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            lag_2.append(new_row)
    df_lag_2 = pd.DataFrame(data=lag_2, columns=columns)

    # Mixed groundwater analysis
    # Repeat end-member splitting, but keep constant values of pdel_bar each season for all years using the summer
    # and winter isotope values averaged over the entire study period
    p_isotopes = []
    p_weights = []
    p_season = []
    for i in range(len(pwt)):
        for d in range(len(pwt[i])):
            p_isotopes.append(pdel[i][d])
            p_weights.append(pwt[i][d])
            p_season.append(pdelcat[i][d])
    mixed = []
    for i in range(len(years)):
        if pwt[i]:
            new_row = endsplit(p_isotopes, qdel[i], p_weights, qwt[i], p_season, qdelcat[i], fluxes[i]['P'],
                        fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            mixed.append(new_row)
    df_mixed = pd.DataFrame(data=mixed, columns=columns)

    # Analysis using precipitation delta values averaged over current and previous year
    merge_years = 1
    merge_pdel_1 = [[] for _ in pwt]
    merge_weights_1 = [[] for _ in pwt]
    merge_season_1 = [[] for _ in pwt]
    for i in range(merge_years, len(pwt)):
        for d in range(len(pwt[i])):
            merge_pdel_1[i].append(pdel[i][d])
            merge_weights_1[i].append(pwt[i][d])
            merge_season_1[i].append(pdelcat[i][d])
        for d in range(len(pwt[i-1])):
            merge_pdel_1[i].append(pdel[i-1][d])
            merge_weights_1[i].append(pwt[i-1][d])
            merge_season_1[i].append(pdelcat[i-1][d])
    lag_1_mean = []
    for i in range(merge_years, len(years)):
        if pwt[i]:
            new_row = endsplit(merge_pdel_1[i], qdel[i], merge_weights_1[i], qwt[i], merge_season_1[i], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            lag_1_mean.append(new_row)
    df_lag_1_mean = pd.DataFrame(data=lag_1_mean, columns=columns)

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
            new_row = endsplit(merge_pdel_2[i], qdel[i], merge_weights_2[i], qwt[i], merge_season_2[i], qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
            new_row[0] = years[i]
            lag_2_mean.append(new_row)
    df_lag_2_mean = pd.DataFrame(data=lag_2_mean, columns=columns)

    df = {"No Lag": df_no_lag, "Lag 1": df_lag_1, "Lag 2": df_lag_2, "Mixed": df_mixed, "Lag 1 Mean": df_lag_1_mean,
            "Lag 2 Mean": df_lag_2_mean}

    # Long-Term Partitioning
    q_isotopes = []
    q_weights = []
    q_season = []
    precipitation = []
    discharge = []
    precipitation_season = []
    discharge_season = []
    for i in range(len(qwt)):
        for d in range(len(qwt[i])):
            q_isotopes.append(qdel[i][d])
            q_weights.append(qwt[i][d])
            q_season.append(qdelcat[i][d])
    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['P'])):
            precipitation.append(fluxes[i]['P'][d])
            discharge.append(fluxes[i]['Q'][d])
            precipitation_season.append(fluxes[i]['Pcat'][d])
            discharge_season.append(fluxes[i]['Qcat'][d])

    lt_list, lt_table = endsplit(p_isotopes, q_isotopes, p_weights, q_weights, p_season, q_season,
                                               precipitation, discharge, precipitation_season, discharge_season)
    print(lt_table)
    count_years = 0
    for i in pwt:
        if i:
            count_years += 1
    lt_list[0] = count_years
    longterm = {columns[i]: lt_list[i] for i in range(len(lt_list))}
    P_s = longterm['P_s']
    P_w = longterm['P_w']
    print('Summer Precipitation:', round(P_s/count_years), '±', round(longterm['P_s_se']/count_years), 'mm with ',
          round((P_s / count_years) * lt_table.loc['ET','eta.summer']), '±',
          round((P_s / count_years) * lt_table.loc['ET', 'eta.summer.se']), 'mm (',
          round(lt_table.loc['ET', 'eta.summer'] * 100), '±', round(lt_table.loc['ET', 'eta.summer.se'] * 100),
          '%) to ET,', round((P_s / count_years) * lt_table.loc['summer', 'eta.summer']),  '±',
          round((P_s / count_years) * lt_table.loc['summer', 'eta.summer.se']), 'mm (',
          round(lt_table.loc['summer', 'eta.summer'] * 100), '±', round(lt_table.loc['summer', 'eta.summer.se'] * 100),
          '%) to summer streamflow, and', round((P_s / count_years) * lt_table.loc['winter', 'eta.summer']), '±',
          round((P_s / count_years) * lt_table.loc['winter', 'eta.summer.se']), 'mm (',
          round(lt_table.loc['winter', 'eta.summer'] * 100), '±', round(lt_table.loc['winter', 'eta.summer.se'] *
                                                                         100), '%) to winter streamflow')
    print('Winter Precipitation: ', round(P_w / count_years), '±', round(longterm['P_w_se'] / count_years), ' mm with',
          round((P_w / count_years) * lt_table.loc['ET', 'eta.winter']), '±',
          round((P_w / count_years) * lt_table.loc['ET', 'eta.winter.se']), 'mm (',
          round(lt_table.loc['ET', 'eta.winter'] * 100), '±', round(lt_table.loc['ET', 'eta.winter.se'] * 100),
          '%) to ET,', round((P_w / count_years) * lt_table.loc['summer', 'eta.winter']), '±',
          round((P_w / count_years) * lt_table.loc['summer', 'eta.winter.se']), 'mm (',
          round(lt_table.loc['summer', 'eta.winter'] * 100), '±', round(lt_table.loc['summer', 'eta.winter.se'] * 100),
          '%) to summer streamflow, and',
          round((P_w / count_years) * lt_table.loc['winter', 'eta.winter']), '±',
          round((P_w / count_years) * lt_table.loc['winter', 'eta.winter.se']), 'mm (',
          round(lt_table.loc['winter', 'eta.winter'] * 100), '±', round(lt_table.loc['winter', 'eta.winter.se'] *
                                                                         100), '%) to winter streamflow')
    print('ET:', round(lt_table.loc['ET', 'f.summer'] * 100), '±', round(lt_table.loc['ET', 'f.summer.se'] * 100),
          '% from summer,', round(lt_table.loc['ET', 'f.winter'] * 100), '±', round(lt_table.loc['ET', 'f.winter.se']
                                                                                     * 100), '% from winter')
    print('Summer Streamflow:', round(lt_table.loc['summer', 'f.summer'] * 100), '±', round(lt_table.loc['summer', 'f.summer.se'] * 100),
          '% from summer,', round(lt_table.loc['summer', 'f.winter'] * 100), '±', round(lt_table.loc['summer', 'f.winter.se'] * 100), '% from winter')
    print('Winter Streamflow:', round(lt_table.loc['winter', 'f.summer'] * 100), '±', round(lt_table.loc['winter', 'f.summer.se'] * 100),
          '% from summer,', round(lt_table.loc['winter', 'f.winter'] * 100), '±', round(lt_table.loc['winter', 'f.winter.se'] * 100),
          '% from winter')
    print('All Streamflow:', round(lt_table.loc['AllQ', 'f.summer'] * 100), '±', round(lt_table.loc['AllQ', 'f.summer.se'] * 100), '% from summer,',
          round(lt_table.loc['AllQ', 'f.winter'] * 100), '±', round(lt_table.loc['AllQ', 'f.winter.se'] * 100),
          '% from winter')

    p_isotopes_elevation = []
    if label != 'Lysimeter':
        if label == 'All':
            for i in range(len(p_isotopes)):
                p_isotopes_elevation.append(p_isotopes[i] - 0.12)
        if label == 'Upper':
            for i in range(len(p_isotopes)):
                p_isotopes_elevation.append(p_isotopes[i] - 0.18)
        elevation_adjusted_list = endsplit(p_isotopes_elevation, q_isotopes, p_weights, q_weights, p_season, q_season,
                                           precipitation, discharge, precipitation_season, discharge_season)[0]
        f_et_change = longterm['f_ET'] - elevation_adjusted_list[8]
        f_ps_change = longterm['f_Ps'] - elevation_adjusted_list[13]
        print('elevation correction decreased fET←PS by', round(f_et_change * 100), '% and fPS→ET by', round(f_ps_change * 100), '% for', label)

    #
    # Decrease streamflow by 1 & 2 permil to estimate effect of evaporative fractionation

    q_isotopes_1 = copy.deepcopy(q_isotopes)
    q_isotopes_2 = copy.deepcopy(q_isotopes)
    for i in range(len(q_isotopes)):
        q_isotopes_1[i] -= 1
        q_isotopes_2[i] -= 2
    new_row = endsplit(p_isotopes, q_isotopes_1, p_weights, q_weights, p_season, q_season,
                                           precipitation, discharge, precipitation_season, discharge_season)[0]
    qdel_minus_1 = {columns[i]: new_row[i] for i in range(len(new_row))}

    new_row = endsplit(p_isotopes, q_isotopes_2, p_weights, q_weights, p_season, q_season,
                                           precipitation, discharge, precipitation_season, discharge_season)[0]
    qdel_minus_2 = {columns[i]: new_row[i] for i in range(len(new_row))}

    print('')
    print('Evaporative Fractionation Sensitivity Analysis')
    print("-1 qdel difference f_ET: ", round((longterm['f_ET'] - qdel_minus_1['f_ET']) * 100))
    print("-1 qdel difference f_Ps: ", round((longterm['f_Ps'] - qdel_minus_1['f_Ps']) * 100))
    print('-2 qdel difference f_ET: ', round((longterm['f_ET'] - qdel_minus_2['f_ET']) * 100))
    print('-2 qdel difference f_Ps: ', round((longterm['f_Ps'] - qdel_minus_2['f_Ps']) * 100))

    # Calculate the potential precipitation undercatch assuming 50% undercatch when snowing and 15% undercatch when raining
    def max_undercatch(flux_data, category='both'):
        if category == 'both':
            for y in range(len(flux_data)):
                for d in range(len(flux_data[y]['P'])):
                    if flux_data[y]['Tcat'][d] == 'snow':
                        flux_data[y]['P'][d] = (flux_data[y]['P'][d] * 1.5)
                    if flux_data[y]['Tcat'][d] == 'rain':
                        flux_data[y]['P'][d] = (flux_data[y]['P'][d] * 1.15)

        if category == 'snow':
            for y in range(len(flux_data)):
                for d in range(len(flux_data[y]['P'])):
                    if flux_data[y]['Tcat'][d] == 'snow':
                        flux_data[y]['P'][d] = (flux_data[y]['P'][d] * 1.5)

        if category == 'rain':
            for y in range(len(flux_data)):
                for d in range(len(flux_data[y]['P'])):
                    if flux_data[y]['Tcat'][d] == 'rain':
                        flux_data[y]['P'][d] = (flux_data[y]['P'][d] * 1.15)

        return flux_data

    # Adjust both rain and snowfall for undercatch

    precip_adjusted_fluxes = max_undercatch(fluxes, category='both')
    pwt, qwt = sum_precipitation_and_runoff(date_by_year, precip_adjusted_fluxes, precip_d_year, runoff_d_year)
    p_weights = []
    for i in range(len(pwt)):
        for d in range(len(pwt[i])):
            p_weights.append(pwt[i][d])
    new_row = endsplit(p_isotopes, q_isotopes, p_weights, q_weights, p_season, q_season,
                                            precipitation, discharge, precipitation_season, discharge_season)[0]
    undercatch_both = {columns[i]: new_row[i] for i in range(len(new_row))}

    # Adjust only rainfall for undercatch

    rain_adjusted_fluxes = max_undercatch(fluxes, category='rain')
    pwt, qwt = sum_precipitation_and_runoff(date_by_year, rain_adjusted_fluxes, precip_d_year, runoff_d_year)
    p_weights = []
    for i in range(len(pwt)):
        for d in range(len(pwt[i])):
            p_weights.append(pwt[i][d])
    new_row = endsplit(p_isotopes, q_isotopes, p_weights, q_weights, p_season, q_season,
                               precipitation, discharge, precipitation_season, discharge_season)[0]
    undercatch_rain = {columns[i]: new_row[i] for i in range(len(new_row))}

    # Adjust only snowfall for undercatch

    snow_adjusted_fluxes = max_undercatch(fluxes, category='snow')
    pwt, qwt = sum_precipitation_and_runoff(date_by_year, snow_adjusted_fluxes, precip_d_year, runoff_d_year)
    p_weights = []
    for i in range(len(pwt)):
        for d in range(len(pwt[i])):
            p_weights.append(pwt[i][d])
    new_row = endsplit(p_isotopes, q_isotopes, p_weights, q_weights, p_season, q_season,
                               precipitation, discharge, precipitation_season, discharge_season)[0]
    undercatch_snow = {columns[i]: new_row[i] for i in range(len(new_row))}

    print('')
    print('Undercatch Sensitivity Analysis')
    print("Adjusted rain difference f_ET: ", round((longterm['f_ET'] - undercatch_rain['f_ET']) * 100))
    print("Adjusted rain difference f_Ps: ", round((longterm['f_Ps'] - undercatch_rain['f_Ps']) * 100))
    print('Adjusted snow difference f_ET: ', round((longterm['f_ET'] - undercatch_snow['f_ET']) * 100))
    print('Adjusted snow difference f_Ps: ', round((longterm['f_Ps'] - undercatch_snow['f_Ps']) * 100))
    print('Adjusted both difference f_ET: ', round((longterm['f_ET'] - undercatch_both['f_ET']) * 100))
    print('Adjusted both difference f_Ps: ', round((longterm['f_Ps'] - undercatch_both['f_Ps']) * 100))

    return df, longterm, qwt_list, qdel_list, qdate_list

def confidence_intervals(df, xlabel='P_s'):
    def make_ci_dict(f_et, f_ps):
        ci_dict = {'xlabel': f_et['xlabel'],
                 'f_ET': f_et['f'],
                 'f_Ps': f_ps['f'],
                 'x_val_ci': f_et['x_val_ci'],
                 'f_ET_fitvals': f_et['fitvals'],
                 'f_ET_ci_low': f_et['ci_low'],
                 'f_ET_ci_upp': f_et['ci_upp'],
                 'f_Ps_fitvals': f_ps['fitvals'],
                 'f_Ps_ci_low': f_ps['ci_low'],
                 'f_Ps_ci_upp': f_ps['ci_upp'],
                 'f_ET_slope': f_et['slope'],
                 'f_ET_slope_pval': f_et['slope p-val'],
                 'f_Ps_slope': f_ps['slope'],
                 'f_Ps_slope_pval': f_ps['slope p-val']}
        return ci_dict

    no_lag_ci_f_et = calculate_wls(df["No Lag"][xlabel], df["No Lag"]['f_ET'], df["No Lag"]['f_ET_se'], x_bound=xlabel)
    no_lag_ci_f_ps = calculate_wls(df["No Lag"][xlabel], df["No Lag"]['f_Ps'], df["No Lag"]['f_Ps_se'], x_bound=xlabel)
    no_lag_ci = make_ci_dict(no_lag_ci_f_et, no_lag_ci_f_ps)

    lag_1_ci_f_et = calculate_wls(df["Lag 1"][xlabel], df["Lag 1"]['f_ET'], df["Lag 1"]['f_ET_se'], x_bound=xlabel)
    lag_1_ci_f_ps = calculate_wls(df["Lag 1"][xlabel], df["Lag 1"]['f_Ps'], df["Lag 1"]['f_Ps_se'], x_bound=xlabel)
    lag_1_ci = make_ci_dict(lag_1_ci_f_et, lag_1_ci_f_ps)

    lag_2_ci_f_et = calculate_wls(df["Lag 2"][xlabel], df["Lag 2"]['f_ET'], df["Lag 2"]['f_ET_se'], x_bound=xlabel)
    lag_2_ci_f_ps = calculate_wls(df["Lag 2"][xlabel], df["Lag 2"]['f_Ps'], df["Lag 2"]['f_Ps_se'], x_bound=xlabel)
    lag_2_ci = make_ci_dict(lag_2_ci_f_et, lag_2_ci_f_ps)

    mixed_ci_f_et = calculate_wls(df["Mixed"][xlabel], df["Mixed"]['f_ET'], df["Mixed"]['f_ET_se'], x_bound=xlabel)
    mixed_ci_f_ps = calculate_wls(df["Mixed"][xlabel], df["Mixed"]['f_Ps'], df["Mixed"]['f_Ps_se'], x_bound=xlabel)
    mixed_ci = make_ci_dict(mixed_ci_f_et, mixed_ci_f_ps)

    lag_1_mean_ci_f_et = calculate_wls(df["Lag 1 Mean"][xlabel], df["Lag 1 Mean"]['f_ET'], df["Lag 1 Mean"]['f_ET_se'], x_bound=xlabel)
    lag_1_mean_ci_f_ps = calculate_wls(df["Lag 1 Mean"][xlabel], df["Lag 1 Mean"]['f_Ps'], df["Lag 1 Mean"]['f_Ps_se'], x_bound=xlabel)
    lag_1_mean_ci = make_ci_dict(lag_1_mean_ci_f_et, lag_1_mean_ci_f_ps)

    lag_2_mean_ci_f_et = calculate_wls(df["Lag 2 Mean"][xlabel], df["Lag 2 Mean"]['f_ET'], df["Lag 2 Mean"]['f_ET_se'], x_bound=xlabel)
    lag_2_mean_ci_f_ps = calculate_wls(df["Lag 2 Mean"][xlabel], df["Lag 2 Mean"]['f_Ps'], df["Lag 2 Mean"]['f_Ps_se'], x_bound=xlabel)
    lag_2_mean_ci = make_ci_dict(lag_2_mean_ci_f_et, lag_2_mean_ci_f_ps)

    df_ci = {"No Lag": no_lag_ci, "Lag 1": lag_1_ci, "Lag 2": lag_2_ci, "Mixed": mixed_ci,
             "Lag 1 Mean": lag_1_mean_ci, "Lag 2 Mean": lag_2_mean_ci}
    return df_ci

# Read date and precipitation data
data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHB.csv')
daily_data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHBDaily.csv')
sampling_dates = data.loc[:, 'date']
precip_isotope = data.loc[:, 'Precip_18O_combined']
interval = data.loc[:, 'time_interval_d']
date_daily = daily_data['date'].to_list()
precip_mm = daily_data['Sum(Precip_mm)'].tolist()
temperature = daily_data['Mean(airtemp_C)'].tolist()

# All Rietholzbach Catchment discharge data
stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
q_all = daily_data['Summed_RHB_discharge_mm'].tolist()

print('Figure 3')
print('All')
df_all, long_term_all, qwt_all, qdel_all, qdate_all = workflow_endsplit(sampling_dates, precip_isotope, interval,
            stream_isotope, date_daily, precip_mm, q_all, temperature, "All")
ci_all_method_1_ps = confidence_intervals(df_all, xlabel='P_s')
ci_all_method_1_pw = confidence_intervals(df_all, xlabel='P_w')
ci_all_method_1_ratio = confidence_intervals(df_all, xlabel='ratio')
wtd_mean_stream_all, error_stream_all = calc_q(q_all, stream_isotope, sampling_dates, date_daily)

# Upper Rietholzbach dicharge data
stream_isotope_upper = data.loc[:, 'Upper_RHB_18O_combined']
q_upper = daily_data['SummedDischarge(upper_RHB_mm/h)'].tolist()
print('')
print('Upper')
df_upper, long_term_upper, qwt_upper, qdel_upper, qdate_upper = workflow_endsplit(sampling_dates, precip_isotope,
            interval, stream_isotope_upper, date_daily, precip_mm, q_upper, temperature, "Upper")
ci_upper_method_1_ps = confidence_intervals(df_upper, xlabel='P_s')
ci_upper_method_1_pw = confidence_intervals(df_upper, xlabel='P_w')
ci_upper_method_1_ratio = confidence_intervals(df_upper, xlabel='ratio')
wtd_mean_stream_upper, error_stream_upper = calc_q(q_upper, stream_isotope_upper, sampling_dates, date_daily)

# Lysimeter seepage data
isotope_lysimeter_seepage = data.loc[:, 'Lysimeter_18O_combined']
amount_lysimeter_seepage = daily_data['lys_seep_mm/day'].tolist()
print('')
print('Lysimeter')
df_lys, long_term_lys, qwt_lys, qdel_lys, qdate_lys = workflow_endsplit(sampling_dates, precip_isotope, interval,
            isotope_lysimeter_seepage, date_daily, precip_mm, amount_lysimeter_seepage, temperature, "Lysimeter")
ci_lys_method_1_ps = confidence_intervals(df_lys, xlabel='P_s')
ci_lys_method_1_pw = confidence_intervals(df_lys, xlabel='P_w')
ci_lys_method_1_ratio = confidence_intervals(df_lys, xlabel='ratio')
wtd_mean_stream_lys, error_stream_lys = calc_q(amount_lysimeter_seepage, isotope_lysimeter_seepage, sampling_dates,
                                              date_daily)

# Figure showing delta values of precipitation and streamflow
wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month = \
    calc_precip(sampling_dates, precip_mm, precip_isotope, interval, 4, 9)
wtd_mean_stream = [wtd_mean_stream_all, wtd_mean_stream_upper, wtd_mean_stream_lys]
error_stream = [error_stream_all, error_stream_upper, error_stream_lys]
stream_label = ["All RHB", np.nan, "Upper RHB and Lysimeter"] # Upper and Lysimeter average delta values overlap on plot
colors = ["blue", "orange", "green"]

date_all = []
pt_size_all = []
for i in range(len(qdate_all)):
    date_all.append(to_year_fraction(qdate_all[i])*11)
    pt_size_all.append(qwt_all[i]*20)
date_upper = []
pt_size_upper = []
for i in range(len(qdate_upper)):
    date_upper.append(to_year_fraction(qdate_upper[i])*11)
    pt_size_upper.append(qwt_upper[i]*20)
date_lys = []
pt_size_lys = []
for i in range(len(qdate_lys)):
    date_lys.append(to_year_fraction(qdate_lys[i])*11)
    pt_size_lys.append(qwt_lys[i]*20)

plot_del_figure(4, 9, wtd_mean_summer, s_error_summer, wtd_mean_winter, s_error_winter, wtd_mean_per_month,
        s_error_per_month, wtd_mean_stream, stream_label, colors, date_all, date_upper, date_lys, qdel_all, qdel_upper,
        qdel_lys, pt_size_all, pt_size_upper, pt_size_lys)

# Weighted Regression of Significant Slopes
x = [1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
     2012, 2013]
count_f_et_method_1_ps, count_f_ps_method_1_ps = plot_panels(ci_all_method_1_ps, ci_upper_method_1_ps, ci_lys_method_1_ps, "Summer Precipitation (mm)",
           'Calculated Using Measured Precipitation, Measured Streamflow, and Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method1ps.eps', dpi=400)
plt.show()
count_f_et_method_1_pw, count_f_ps_method_1_pw = plot_panels(ci_all_method_1_pw, ci_upper_method_1_pw, ci_lys_method_1_pw, "Winter Precipitation (mm)",
           'Calculated Using Measured Precipitation, Measured Streamflow, and Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method1pw.eps', dpi=400)
plt.show()
count_f_et_method_1_ratio, count_f_ps_method_1_ratio = plot_panels(ci_all_method_1_ratio, ci_upper_method_1_ratio, ci_lys_method_1_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Measured Precipitation, Measured Streamflow, and Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method1ratio.eps', dpi=400)
plt.show()

# Figure showing evapotranspiration from lysimeter weights
evapotranspiration = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\Evapotranspiration_Hirschi.csv')

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

df_all_method_2 = {"No Lag":calculate_fractions(df_all_undercatch_adj['No Lag'], et='mass bal'),
                   "Lag 1": calculate_fractions(df_all_undercatch_adj['Lag 1'], et='mass bal'),
                   "Lag 2": calculate_fractions(df_all_undercatch_adj['Lag 2'], et='mass bal'),
                   "Mixed": calculate_fractions(df_all_undercatch_adj['Mixed'], et='mass bal'),
                   "Lag 1 Mean": calculate_fractions(df_all_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                   "Lag 2 Mean": calculate_fractions(df_all_undercatch_adj['Lag 2 Mean'], et='mass bal')}
ci_all_method_2_ps = confidence_intervals(df_all_method_2, xlabel='P_s')
ci_all_method_2_pw = confidence_intervals(df_all_method_2, xlabel='P_w')
ci_all_method_2_ratio = confidence_intervals(df_all_method_2, xlabel='ratio')

df_upper_method_2 = {"No Lag": calculate_fractions(df_upper_undercatch_adj['No Lag'], et='mass bal'),
                    "Lag 1": calculate_fractions(df_upper_undercatch_adj['Lag 1'], et='mass bal'),
                    "Lag 2": calculate_fractions(df_upper_undercatch_adj['Lag 2'], et='mass bal'),
                    "Mixed": calculate_fractions(df_upper_undercatch_adj['Mixed'], et='mass bal'),
                    "Lag 1 Mean": calculate_fractions(df_upper_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                    "Lag 2 Mean": calculate_fractions(df_upper_undercatch_adj['Lag 2 Mean'], et='mass bal')}
ci_upper_method_2_ps = confidence_intervals(df_upper_method_2, xlabel='P_s')
ci_upper_method_2_pw = confidence_intervals(df_upper_method_2, xlabel='P_w')
ci_upper_method_2_ratio = confidence_intervals(df_upper_method_2, xlabel='ratio')

df_lys_method_2 = {"No Lag": calculate_fractions(df_lys_undercatch_adj['No Lag'], et='mass bal'),
                   "Lag 1": calculate_fractions(df_lys_undercatch_adj['Lag 1'], et='mass bal'),
                   "Lag 2": calculate_fractions(df_lys_undercatch_adj['Lag 2'], et='mass bal'),
                   "Mixed": calculate_fractions(df_lys_undercatch_adj['Mixed'], et='mass bal'),
                   "Lag 1 Mean": calculate_fractions(df_lys_undercatch_adj['Lag 1 Mean'], et='mass bal'),
                   "Lag 2 Mean": calculate_fractions(df_lys_undercatch_adj['Lag 2 Mean'], et='mass bal')}
ci_lys_method_2_ps = confidence_intervals(df_lys_method_2, xlabel='P_s')
ci_lys_method_2_pw = confidence_intervals(df_lys_method_2, xlabel='P_w')
ci_lys_method_2_ratio = confidence_intervals(df_lys_method_2, xlabel='ratio')

count_f_et_method_2_ps, count_f_ps_method_2_ps = plot_panels(ci_all_method_2_ps, ci_upper_method_2_ps, ci_lys_method_2_ps, "Summer Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Measured Streamflow, and Annually Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method2ps.eps', dpi=400)
plt.show()
count_f_et_method_2_pw, count_f_ps_method_2_pw = plot_panels(ci_all_method_2_pw, ci_upper_method_2_pw, ci_lys_method_2_pw, "Winter Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Measured Streamflow, and Annually Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method2pw.eps', dpi=400)
plt.show()
count_f_et_method_2_ratio, count_f_ps_method_2_ratio = plot_panels(ci_all_method_2_ratio, ci_upper_method_2_ratio, ci_lys_method_2_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Undercatch-Adjusted Precipitation, Measured Streamflow, and Annually Mass-Balanced ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method2ratio.eps', dpi=400)
plt.show()

df_all_method_3 = {"No Lag":calculate_fractions(calculate_scaled_et(df_all['No Lag'], evapotranspiration)),
                   "Lag 1": calculate_fractions(calculate_scaled_et(df_all['Lag 1'], evapotranspiration)),
                   "Lag 2": calculate_fractions(calculate_scaled_et(df_all['Lag 2'], evapotranspiration)),
                   "Mixed": calculate_fractions(calculate_scaled_et(df_all['Mixed'], evapotranspiration)),
                   "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_all['Lag 1 Mean'], evapotranspiration)),
                   "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_all['Lag 2 Mean'], evapotranspiration))}
ci_all_method_3_ps = confidence_intervals(df_all_method_3, xlabel='P_s')
ci_all_method_3_pw = confidence_intervals(df_all_method_3, xlabel='P_w')
ci_all_method_3_ratio = confidence_intervals(df_all_method_3, xlabel='ratio')

df_upper_method_3 = {"No Lag": calculate_fractions(calculate_scaled_et(df_upper['No Lag'], evapotranspiration)),
                     "Lag 1": calculate_fractions(calculate_scaled_et(df_upper['Lag 1'], evapotranspiration)),
                     "Lag 2": calculate_fractions(calculate_scaled_et(df_upper['Lag 2'], evapotranspiration)),
                     "Mixed": calculate_fractions(calculate_scaled_et(df_upper['Mixed'], evapotranspiration)),
                     "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_upper['Lag 1 Mean'], evapotranspiration)),
                     "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_upper['Lag 2 Mean'], evapotranspiration))}
ci_upper_method_3_ps = confidence_intervals(df_upper_method_3, xlabel='P_s')
ci_upper_method_3_pw = confidence_intervals(df_upper_method_3, xlabel='P_w')
ci_upper_method_3_ratio = confidence_intervals(df_upper_method_3, xlabel='ratio')

df_lys_method_3 = {"No Lag": calculate_fractions(calculate_scaled_et(df_lys['No Lag'], evapotranspiration)),
                   "Lag 1": calculate_fractions(calculate_scaled_et(df_lys['Lag 1'], evapotranspiration)),
                   "Lag 2": calculate_fractions(calculate_scaled_et(df_lys['Lag 2'], evapotranspiration)),
                   "Mixed": calculate_fractions(calculate_scaled_et(df_lys['Mixed'], evapotranspiration)),
                   "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_lys['Lag 1 Mean'], evapotranspiration)),
                   "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_lys['Lag 2 Mean'], evapotranspiration))}
ci_lys_method_3_ps = confidence_intervals(df_lys_method_3, xlabel='P_s')
ci_lys_method_3_pw = confidence_intervals(df_lys_method_3, xlabel='P_w')
ci_lys_method_3_ratio = confidence_intervals(df_lys_method_3, xlabel='ratio')

count_f_et_method_3_ps, count_f_ps_method_3_ps = plot_panels(ci_all_method_3_ps, ci_upper_method_3_ps, ci_lys_method_3_ps, "Summer Precipitation (mm)",
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method3ps.eps', dpi=400)
plt.show()
count_f_et_method_3_pw, count_f_ps_method_3_pw = plot_panels(ci_all_method_3_pw, ci_upper_method_3_pw, ci_lys_method_3_pw, "Winter Precipitation (mm)",
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method3pw.eps', dpi=400)
plt.show()
count_f_et_method_3_ratio, count_f_ps_method_3_ratio = plot_panels(ci_all_method_3_ratio, ci_upper_method_3_ratio, ci_lys_method_3_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method3ratio.eps', dpi=400)
plt.show()

df_all_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_all_undercatch_adj['Lag 2 Mean'], evapotranspiration))}
ci_all_method_4_ps = confidence_intervals(df_all_method_4, xlabel='P_s')
ci_all_method_4_pw = confidence_intervals(df_all_method_4, xlabel='P_w')
ci_all_method_4_ratio = confidence_intervals(df_all_method_4, xlabel='ratio')

df_upper_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean":calculate_fractions(calculate_scaled_et(df_upper_undercatch_adj['Lag 2 Mean'], evapotranspiration))}
ci_upper_method_4_ps = confidence_intervals(df_upper_method_4, xlabel='P_s')
ci_upper_method_4_pw = confidence_intervals(df_upper_method_4, xlabel='P_w')
ci_upper_method_4_ratio = confidence_intervals(df_upper_method_4, xlabel='ratio')

df_lys_method_4 = {"No Lag": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['No Lag'], evapotranspiration)),
        "Lag 1": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 1'], evapotranspiration)),
        "Lag 2": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 2'], evapotranspiration)),
        "Mixed": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Mixed'], evapotranspiration)),
        "Lag 1 Mean": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 1 Mean'], evapotranspiration)),
        "Lag 2 Mean": calculate_fractions(calculate_scaled_et(df_lys_undercatch_adj['Lag 2 Mean'], evapotranspiration))}
ci_lys_method_4_ps = confidence_intervals(df_lys_method_4, xlabel='P_s')
ci_lys_method_4_pw = confidence_intervals(df_lys_method_4, xlabel='P_w')
ci_lys_method_4_ratio = confidence_intervals(df_lys_method_4, xlabel='ratio')

count_f_et_method_4_ps, count_f_ps_method_4_ps = plot_panels(ci_all_method_4_ps, ci_upper_method_4_ps, ci_lys_method_4_ps, "Summer Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method4ps.eps', dpi=400)
plt.show()
count_f_et_method_4_pw, count_f_ps_method_4_pw = plot_panels(ci_all_method_4_pw, ci_upper_method_4_pw, ci_lys_method_4_pw, "Winter Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method4pw.eps', dpi=400)
plt.show()
count_f_et_method_4_ratio, count_f_ps_method_4_ratio = plot_panels(ci_all_method_4_ratio, ci_upper_method_4_ratio, ci_lys_method_4_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Lysimeter-Scaled ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method4ratio.eps', dpi=400)
plt.show()

df_all_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_all['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_all['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_all['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_all['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_all['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_all['Lag 2 Mean']))}
ci_all_method_5_ps = confidence_intervals(df_all_method_5, xlabel='P_s')
ci_all_method_5_pw = confidence_intervals(df_all_method_5, xlabel='P_w')
ci_all_method_5_ratio = confidence_intervals(df_all_method_5, xlabel='ratio')

df_upper_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_upper['No Lag'])),
                     "Lag 1": calculate_fractions(calculate_avg_et(df_upper['Lag 1'])),
                     "Lag 2": calculate_fractions(calculate_avg_et(df_upper['Lag 2'])),
                     "Mixed": calculate_fractions(calculate_avg_et(df_upper['Mixed'])),
                     "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_upper['Lag 1 Mean'])),
                     "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_upper['Lag 2 Mean']))}
ci_upper_method_5_ps = confidence_intervals(df_upper_method_5, xlabel='P_s')
ci_upper_method_5_pw = confidence_intervals(df_upper_method_5, xlabel='P_w')
ci_upper_method_5_ratio = confidence_intervals(df_upper_method_5, xlabel='ratio')

df_lys_method_5 = {"No Lag": calculate_fractions(calculate_avg_et(df_lys['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_lys['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_lys['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_lys['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_lys['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_lys['Lag 2 Mean']))}
ci_lys_method_5_ps = confidence_intervals(df_lys_method_5, xlabel='P_s')
ci_lys_method_5_pw = confidence_intervals(df_lys_method_5, xlabel='P_w')
ci_lys_method_5_ratio = confidence_intervals(df_lys_method_5, xlabel='ratio')

count_f_et_method_5_ps, count_f_ps_method_5_ps = plot_panels(ci_all_method_5_ps, ci_upper_method_5_ps, ci_lys_method_5_ps, "Summer Precipitation (mm)",
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method5ps.eps', dpi=400)
plt.show()
count_f_et_method_5_pw, count_f_ps_method_5_pw = plot_panels(ci_all_method_5_pw, ci_upper_method_5_pw, ci_lys_method_5_pw, "Winter Precipitation (mm)",
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method5pw.eps', dpi=400)
plt.show()
count_f_et_method_5_ratio, count_f_ps_method_5_ratio = plot_panels(ci_all_method_5_ratio, ci_upper_method_5_ratio, ci_lys_method_5_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Measured Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method5ratio.eps', dpi=400)
plt.show()

df_all_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_all_undercatch_adj['Lag 2 Mean']))}
ci_all_method_6_ps = confidence_intervals(df_all_method_6, xlabel='P_s')
ci_all_method_6_pw = confidence_intervals(df_all_method_6, xlabel='P_w')
ci_all_method_6_ratio = confidence_intervals(df_all_method_6, xlabel='ratio')

df_upper_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['No Lag'])),
                     "Lag 1": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 1'])),
                     "Lag 2": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 2'])),
                     "Mixed": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Mixed'])),
                     "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 1 Mean'])),
                     "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_upper_undercatch_adj['Lag 2 Mean']))}
ci_upper_method_6_ps = confidence_intervals(df_upper_method_6, xlabel='P_s')
ci_upper_method_6_pw = confidence_intervals(df_upper_method_6, xlabel='P_w')
ci_upper_method_6_ratio = confidence_intervals(df_upper_method_6, xlabel='ratio')

df_lys_method_6 = {"No Lag": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['No Lag'])),
                   "Lag 1": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 1'])),
                   "Lag 2": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 2'])),
                   "Mixed": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Mixed'])),
                   "Lag 1 Mean": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 1 Mean'])),
                   "Lag 2 Mean": calculate_fractions(calculate_avg_et(df_lys_undercatch_adj['Lag 2 Mean']))}
ci_lys_method_6_ps = confidence_intervals(df_lys_method_6, xlabel='P_s')
ci_lys_method_6_pw = confidence_intervals(df_lys_method_6, xlabel='P_w')
ci_lys_method_6_ratio = confidence_intervals(df_lys_method_6, xlabel='ratio')

count_f_et_method_6_ps, count_f_ps_method_6_ps = plot_panels(ci_all_method_6_ps, ci_upper_method_6_ps, ci_lys_method_6_ps, "Summer Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method6ps.eps', dpi=400)
plt.show()
count_f_et_method_6_pw, count_f_ps_method_6_pw = plot_panels(ci_all_method_6_pw, ci_upper_method_6_pw, ci_lys_method_6_pw, "Winter Precipitation (mm)",
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method6pw.eps', dpi=400)
plt.show()
count_f_et_method_6_ratio, count_f_ps_method_6_ratio = plot_panels(ci_all_method_6_ratio, ci_upper_method_6_ratio, ci_lys_method_6_ratio,
            ('\n'.join(wrap('Ratio of Summer to Winter Precipitation (unitless)', 28))),
           'Calculated Using Undercatch-Adjusted Precipitation, Annually Mass-Balanced Streamflow, and Average ET')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\method6ratio.eps', dpi=400)
plt.show()


table1 = pd.DataFrame(np.zeros((14, 18)))
measured_precip = {'f_et': {'ps': [count_f_et_method_1_ps, count_f_et_method_3_ps, count_f_et_method_5_ps],
                   'pw': [count_f_et_method_1_pw, count_f_et_method_3_pw, count_f_et_method_5_pw],
                    'ratio': [count_f_et_method_1_ratio, count_f_et_method_3_ratio, count_f_et_method_5_ratio]},
                   'f_ps': {'ps': [count_f_ps_method_1_ps, count_f_ps_method_3_ps, count_f_ps_method_5_ps],
                   'pw': [count_f_ps_method_1_pw, count_f_ps_method_3_pw, count_f_ps_method_5_pw],
                    'ratio': [count_f_ps_method_1_ratio, count_f_ps_method_3_ratio, count_f_ps_method_5_ratio]}}
adjusted_precip = {'f_et': {'ps': [count_f_et_method_2_ps, count_f_et_method_4_ps, count_f_et_method_6_ps],
                   'pw': [count_f_et_method_2_pw, count_f_et_method_4_pw, count_f_et_method_6_pw],
                    'ratio': [count_f_et_method_2_ratio, count_f_et_method_4_ratio, count_f_et_method_6_ratio]},
                   'f_ps': {'ps': [count_f_ps_method_2_ps, count_f_ps_method_4_ps, count_f_ps_method_6_ps],
                   'pw': [count_f_ps_method_2_pw, count_f_ps_method_4_pw, count_f_ps_method_6_pw],
                    'ratio': [count_f_ps_method_2_ratio, count_f_ps_method_4_ratio, count_f_ps_method_6_ratio]}}
for j, i in enumerate([measured_precip, adjusted_precip]):
    for item in i['f_et']['ps']:
        table1.iloc[j, 0] += item['TotalCount']['+']
        table1.iloc[j, 1] += item['TotalCount']['-']
        table1.iloc[j, 2] += item['TotalCount']['NA']
    for item in i['f_et']['pw']:
        table1.iloc[j, 3] += item['TotalCount']['+']
        table1.iloc[j, 4] += item['TotalCount']['-']
        table1.iloc[j, 5] += item['TotalCount']['NA']
    for item in i['f_et']['ratio']:
        table1.iloc[j, 6] += item['TotalCount']['+']
        table1.iloc[j, 7] += item['TotalCount']['-']
        table1.iloc[j, 8] += item['TotalCount']['NA']
    for item in i['f_ps']['ps']:
        table1.iloc[j, 9] += item['TotalCount']['+']
        table1.iloc[j, 10] += item['TotalCount']['-']
        table1.iloc[j, 11] += item['TotalCount']['NA']
    for item in i['f_ps']['pw']:
        table1.iloc[j, 12] += item['TotalCount']['+']
        table1.iloc[j, 13] += item['TotalCount']['-']
        table1.iloc[j, 14] += item['TotalCount']['NA']
    for item in i['f_ps']['ratio']:
        table1.iloc[j, 15] += item['TotalCount']['+']
        table1.iloc[j, 16] += item['TotalCount']['-']
        table1.iloc[j, 17] += item['TotalCount']['NA']

mass_bal_et = {'f_et': {'ps': [count_f_et_method_1_ps, count_f_et_method_2_ps],
                   'pw': [count_f_et_method_1_pw, count_f_et_method_2_pw],
                    'ratio': [count_f_et_method_1_ratio, count_f_et_method_2_ratio]},
               'f_ps': {'ps': [count_f_ps_method_1_ps, count_f_ps_method_2_ps],
                   'pw': [count_f_ps_method_1_pw, count_f_ps_method_2_pw],
                    'ratio': [count_f_ps_method_1_ratio, count_f_ps_method_2_ratio]}}
lys_scaled_et = {'f_et': {'ps': [count_f_et_method_3_ps, count_f_et_method_4_ps],
                   'pw': [count_f_et_method_3_pw, count_f_et_method_4_pw],
                    'ratio': [count_f_et_method_3_ratio, count_f_et_method_4_ratio]},
                 'f_ps': {'ps': [count_f_ps_method_3_ps, count_f_ps_method_4_ps],
                   'pw': [count_f_ps_method_3_pw, count_f_ps_method_4_pw],
                    'ratio': [count_f_ps_method_3_ratio, count_f_ps_method_4_ratio]}}
avg_et = {'f_et': {'ps': [count_f_et_method_5_ps, count_f_et_method_6_ps],
                   'pw': [count_f_et_method_5_pw, count_f_et_method_6_pw],
                    'ratio': [count_f_et_method_5_ratio, count_f_et_method_6_ratio]},
          'f_ps': {'ps': [count_f_ps_method_5_ps, count_f_ps_method_6_ps],
                   'pw': [count_f_ps_method_5_pw, count_f_ps_method_6_pw],
                    'ratio': [count_f_ps_method_5_ratio, count_f_ps_method_6_ratio]}}
for j, i in enumerate([mass_bal_et, lys_scaled_et, avg_et]):
    for item in i['f_et']['ps']:
        table1.iloc[j + 2, 0] += item['TotalCount']['+']
        table1.iloc[j + 2, 1] += item['TotalCount']['-']
        table1.iloc[j + 2, 2] += item['TotalCount']['NA']
    for item in i['f_et']['pw']:
        table1.iloc[j + 2, 3] += item['TotalCount']['+']
        table1.iloc[j + 2, 4] += item['TotalCount']['-']
        table1.iloc[j + 2, 5] += item['TotalCount']['NA']
    for item in i['f_et']['ratio']:
        table1.iloc[j + 2, 6] += item['TotalCount']['+']
        table1.iloc[j + 2, 7] += item['TotalCount']['-']
        table1.iloc[j + 2, 8] += item['TotalCount']['NA']
    for item in i['f_ps']['ps']:
        table1.iloc[j + 2, 9] += item['TotalCount']['+']
        table1.iloc[j + 2, 10] += item['TotalCount']['-']
        table1.iloc[j + 2, 11] += item['TotalCount']['NA']
    for item in i['f_ps']['pw']:
        table1.iloc[j + 2, 12] += item['TotalCount']['+']
        table1.iloc[j + 2, 13] += item['TotalCount']['-']
        table1.iloc[j + 2, 14] += item['TotalCount']['NA']
    for item in i['f_ps']['ratio']:
        table1.iloc[j + 2, 15] += item['TotalCount']['+']
        table1.iloc[j + 2, 16] += item['TotalCount']['-']
        table1.iloc[j + 2, 17] += item['TotalCount']['NA']

every_method = {'f_et': {'ps': [count_f_et_method_1_ps, count_f_et_method_2_ps, count_f_et_method_3_ps, count_f_et_method_4_ps,
                       count_f_et_method_5_ps, count_f_et_method_6_ps],
                'pw': [count_f_et_method_1_pw, count_f_et_method_2_pw, count_f_et_method_3_pw, count_f_et_method_4_pw,
                       count_f_et_method_5_pw, count_f_et_method_6_pw],
             'ratio': [count_f_et_method_1_ratio, count_f_et_method_2_ratio, count_f_et_method_3_ratio, 
                       count_f_et_method_4_ratio, count_f_et_method_5_ratio, count_f_et_method_6_ratio]},
                'f_ps': {'ps': [count_f_ps_method_1_ps, count_f_ps_method_2_ps, count_f_ps_method_3_ps, count_f_ps_method_4_ps,
                       count_f_ps_method_5_ps, count_f_ps_method_6_ps],
                'pw': [count_f_ps_method_1_pw, count_f_ps_method_2_pw, count_f_ps_method_3_pw, count_f_ps_method_4_pw,
                       count_f_ps_method_5_pw, count_f_ps_method_6_pw],
             'ratio': [count_f_ps_method_1_ratio, count_f_ps_method_2_ratio, count_f_ps_method_3_ratio, 
                       count_f_ps_method_4_ratio, count_f_ps_method_5_ratio, count_f_ps_method_6_ratio]}}

for j, i in enumerate(['All RHB', 'Upper RHB', 'Lysimeter']):
    for item in every_method['f_et']['ps']:
        table1.iloc[j + 5, 0] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 1] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 2] += item['ByWatershedCount'][i]['NA']
    for item in every_method['f_et']['pw']:
        table1.iloc[j + 5, 3] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 4] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 5] += item['ByWatershedCount'][i]['NA']
    for item in every_method['f_et']['ratio']:
        table1.iloc[j + 5, 6] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 7] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 8] += item['ByWatershedCount'][i]['NA']
    for item in every_method['f_ps']['ps']:
        table1.iloc[j + 5, 9] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 10] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 11] += item['ByWatershedCount'][i]['NA']
    for item in every_method['f_ps']['pw']:
        table1.iloc[j + 5, 12] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 13] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 14] += item['ByWatershedCount'][i]['NA']
    for item in every_method['f_ps']['ratio']:
        table1.iloc[j + 5, 15] += item['ByWatershedCount'][i]['+']
        table1.iloc[j + 5, 16] += item['ByWatershedCount'][i]['-']
        table1.iloc[j + 5, 17] += item['ByWatershedCount'][i]['NA']

for j, i in enumerate(['No Lag', 'Lag 1', 'Lag 2', 'Mixed', 'Lag 1 Mean', 'Lag 2 Mean']):
    for item in every_method['f_et']['ps']:
        table1.iloc[j + 8, 0] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 1] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 2] += item['ByMethodCount'][i]['NA']
    for item in every_method['f_et']['pw']:
        table1.iloc[j + 8, 3] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 4] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 5] += item['ByMethodCount'][i]['NA']
    for item in every_method['f_et']['ratio']:
        table1.iloc[j + 8, 6] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 7] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 8] += item['ByMethodCount'][i]['NA']
    for item in every_method['f_ps']['ps']:
        table1.iloc[j + 8, 9] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 10] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 11] += item['ByMethodCount'][i]['NA']
    for item in every_method['f_ps']['pw']:
        table1.iloc[j + 8, 12] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 13] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 14] += item['ByMethodCount'][i]['NA']
    for item in every_method['f_ps']['ratio']:
        table1.iloc[j + 8, 15] += item['ByMethodCount'][i]['+']
        table1.iloc[j + 8, 16] += item['ByMethodCount'][i]['-']
        table1.iloc[j + 8, 17] += item['ByMethodCount'][i]['NA']

table1.columns = ['f_et_v_ps_+', 'f_et_v_ps_-', 'f_et_v_ps_ns', 'f_et_v_pw_+', 'f_et_v_pw_-',
                               'f_et_v_pw_ns', 'f_et_v_ratio_+', 'f_et_v_ratio_-', 'f_et_v_ratio_ns',
                               'f_ps_v_ps_+', 'f_ps_v_ps_-', 'f_ps_v_ps_ns', 'f_ps_v_pw_+', 'f_ps_v_pw_-',
                               'f_ps_v_pw_ns', 'f_ps_v_ratio_+', 'f_ps_v_ratio_-', 'f_ps_v_ratio_ns']
table1.index = ['measured_precip', 'adjusted_precip', 'mass_bal_et', 'lys_scaled_et', 'avg_et',
                        'All', 'Upper', 'Lysimeter', 'No Lag', 'Lag 1', 'Lag 2', 'Mixed', 'Lag 1 Mean', 'Lag 2 Mean']
print('')
print('No longer included')
print(table1)

table2 = pd.DataFrame(np.zeros((33, 18)))

for j, i in enumerate([measured_precip, adjusted_precip, mass_bal_et, lys_scaled_et, avg_et]):
    for k, m in enumerate(['All RHB', 'Upper RHB', 'Lysimeter']):
        for item in i['f_et']['ps']:
            table2.iloc[k + j * 3, 0] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 1] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 2] += item['ByWatershedCount'][m]['NA']
        for item in i['f_et']['pw']:
            table2.iloc[k + j * 3, 3] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 4] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 5] += item['ByWatershedCount'][m]['NA']
        for item in i['f_et']['ratio']:
            table2.iloc[k + j * 3, 6] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 7] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 8] += item['ByWatershedCount'][m]['NA']
        for item in i['f_ps']['ps']:
            table2.iloc[k + j * 3, 9] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 10] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 11] += item['ByWatershedCount'][m]['NA']
        for item in i['f_ps']['pw']:
            table2.iloc[k + j * 3, 12] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 13] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 14] += item['ByWatershedCount'][m]['NA']
        for item in i['f_ps']['ratio']:
            table2.iloc[k + j * 3, 15] += item['ByWatershedCount'][m]['+']
            table2.iloc[k + j * 3, 16] += item['ByWatershedCount'][m]['-']
            table2.iloc[k + j * 3, 17] += item['ByWatershedCount'][m]['NA']
for j, i in enumerate(['All RHB', 'Upper RHB', 'Lysimeter']):
    for k, m in enumerate(['No Lag', 'Lag 1', 'Lag 2', 'Mixed', 'Lag 1 Mean', 'Lag 2 Mean']):
        for item in every_method['f_et']['ps']:
            table2.iloc[15 + k + 6 * j, 0] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 1] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 2] += item['ByWatershedandMethod'][i][m]['NA']
        for item in every_method['f_et']['pw']:
            table2.iloc[15 + k + 6 * j, 3] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 4] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 5] += item['ByWatershedandMethod'][i][m]['NA']
        for item in every_method['f_et']['ratio']:
            table2.iloc[15 + k + 6 * j, 6] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 7] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 8] += item['ByWatershedandMethod'][i][m]['NA']
        for item in every_method['f_ps']['ps']:
            table2.iloc[15 + k + 6 * j, 9] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 10] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 11] += item['ByWatershedandMethod'][i][m]['NA']
        for item in every_method['f_ps']['pw']:
            table2.iloc[15 + k + 6 * j, 12] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 13] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 14] += item['ByWatershedandMethod'][i][m]['NA']
        for item in every_method['f_ps']['ratio']:
            table2.iloc[15 + k + 6 * j, 15] += item['ByWatershedandMethod'][i][m]['+']
            table2.iloc[15 + k + 6 * j, 16] += item['ByWatershedandMethod'][i][m]['-']
            table2.iloc[15 + k + 6 * j, 17] += item['ByWatershedandMethod'][i][m]['NA']
print('Table1')
print(table2)

print('')
print('Text')
print('long-term mean P varied from', round(long_term_all['Ptot']/long_term_all['Year']), ' mm y-1 (All RHB,',
      df_all['No Lag']['Year'][0], 'to ', df_all['No Lag']['Year'][0] + long_term_all['Year'],
      ') to', round(long_term_upper['Ptot']/long_term_upper['Year']), 'mm y-1 (Upper RHB,', df_upper['No Lag']['Year'][0],
      'to', df_upper['No Lag']['Year'][0] + long_term_upper['Year'], ') to',
      round(long_term_lys['Ptot']/long_term_lys['Year']), 'mm y-1 (Lysimeter,', df_lys['No Lag']['Year'][0], 'to',
      df_lys['No Lag']['Year'][0] + long_term_lys['Year'], ')')
print('Q was', round(long_term_all['Q']/long_term_all['Year']), 'mm y-1 for All RHB,',  round(long_term_upper['Q']/long_term_upper['Year']),
      'mm y-1 for Upper RHB, and', round(long_term_lys['Q']/long_term_lys['Year']), 'mm y-1 for the lysimeter')
print('long-term mean ET was', round(long_term_all['ET']/long_term_all['Year']), 'mm y-1 for All RHB,', round(long_term_upper['ET']/long_term_upper['Year']),
      'mm y-1 for Upper RHB, and', round(long_term_lys['ET']/long_term_lys['Year']), 'mm y-1 for the lysimeter')
print('The fraction of ET sourced from summer precipitation (fET←PS)  increased with summer precipitation amount for',
      round(table1.loc['All', 'f_et_v_ps_+'] * 100 / 36), '%,', round(table1.loc['Upper', 'f_et_v_ps_+'] * 100 / 36),
      '%, and', round(table1.loc['Lysimeter', 'f_et_v_ps_+'] * 100 / 36), '% of the various fET←PS calculations for',
                                                                           'All, Upper, and Lysimeter, respectively;')
print('in contrast, only', round(table1.loc['All', 'f_et_v_ps_-'] * 100 / 36), '%,', round(table1.loc['Upper', 'f_et_v_ps_-'] * 100 / 36),
        '%, and', round(table1.loc['Lysimeter', 'f_et_v_ps_-'] * 100 / 36), '% showed negative relationships with summer precipitation amount,')
print('and', round(table1.loc['All', 'f_et_v_ps_ns'] * 100 / 36), '%,', round(table1.loc['Upper', 'f_et_v_ps_ns'] * 100 / 36), '%, and', round(table1.loc['Lysimeter', 'f_et_v_ps_ns'] * 100 / 36), '% did not show any significant relationship.')
print('')
print('fET←PS decreased with winter precipitation for', round(table1.loc['All', 'f_et_v_pw_-'] * 100 / 36), '%', round(table1.loc['Upper', 'f_et_v_pw_-'] * 100 / 36), '%',
      round(table1.loc['Lysimeter', 'f_et_v_pw_-'] * 100 / 36), '%, increased with winter precipitation for', round(table1.loc['All', 'f_et_v_pw_+'] * 100 / 36),
    round(table1.loc['Upper', 'f_et_v_pw_+'] * 100 / 36), round(table1.loc['Lysimeter', 'f_et_v_pw_+'] * 100 / 36), 'and non-significant for',
    round(table1.loc['All', 'f_et_v_pw_ns'] * 100 / 36), round(table1.loc['Upper', 'f_et_v_pw_ns'] * 100 / 36), round(table1.loc['Lysimeter', 'f_et_v_pw_ns'] * 100 / 36))
print('')
print('fET←PS increased with the ratio of summer to winter precipitation for', round(table1.loc['All', 'f_et_v_ratio_+'] * 100 / 36), '%', round(table1.loc['Upper', 'f_et_v_ratio_+'] * 100 / 36), '%',
      round(table1.loc['Lysimeter', 'f_et_v_ratio_+'] * 100 / 36), '%, increased with the ratio of summer to winter precipitation for', round(table1.loc['All', 'f_et_v_ratio_-'] * 100 / 36),
    round(table1.loc['Upper', 'f_et_v_ratio_-'] * 100 / 36), round(table1.loc['Lysimeter', 'f_et_v_ratio_-'] * 100 / 36), 'and non-significant for',
    round(table1.loc['All', 'f_et_v_ratio_ns'] * 100 / 36), round(table1.loc['Upper', 'f_et_v_ratio_ns'] * 100 / 36), round(table1.loc['Lysimeter', 'f_et_v_ratio_ns'] * 100 / 36))
print('')
print('fPs to ET increased with summer precipitation for', round(table1.loc['All', 'f_ps_v_ps_+'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_ps_+'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_ps_+'] * 100 / 36), 'decreased for', round(table1.loc['All', 'f_ps_v_ps_-'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_ps_-'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_ps_-'] * 100 / 36), 'non-significant for', round(table1.loc['All', 'f_ps_v_ps_ns'] * 100 / 36),
    round(table1.loc['Upper', 'f_ps_v_ps_ns'] * 100 / 36), round(table1.loc['Lysimeter', 'f_ps_v_ps_ns'] * 100 / 36))
print('')
print('fPs to ET decreased with winter precipitation for', round(table1.loc['All', 'f_ps_v_pw_-'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_pw_-'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_pw_-'] * 100 / 36), 'increased for', round(table1.loc['All', 'f_ps_v_pw_+'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_pw_+'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_pw_+'] * 100 / 36), 'non-significant for', round(table1.loc['All', 'f_ps_v_pw_ns'] * 100 / 36),
    round(table1.loc['Upper', 'f_ps_v_pw_ns'] * 100 / 36), round(table1.loc['Lysimeter', 'f_ps_v_pw_ns'] * 100 / 36))
print('')
print('fPs to ET increased with the ratio of summer to winter precipitation for', round(table1.loc['All', 'f_ps_v_ratio_+'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_ratio_+'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_ratio_+'] * 100 / 36), 'decreased for', round(table1.loc['All', 'f_ps_v_ratio_-'] * 100 / 36), round(table1.loc['Upper', 'f_ps_v_ratio_-'] * 100 / 36),
    round(table1.loc['Lysimeter', 'f_ps_v_ratio_-'] * 100 / 36), 'non-significant for', round(table1.loc['All', 'f_ps_v_ratio_ns'] * 100 / 36),
    round(table1.loc['Upper', 'f_ps_v_ratio_ns'] * 100 / 36), round(table1.loc['Lysimeter', 'f_ps_v_ratio_ns'] * 100 / 36))
print('')
print('Adjustments to ET amount increased the number of positive trends for fET←PS v. PS (', round((table1.loc['lys_scaled_et',
            'f_et_v_ps_+'] - table1.loc['mass_bal_et', 'f_et_v_ps_+']) * 100 / 36), '% and', round((table1.loc['avg_et',
            'f_et_v_ps_+'] - table1.loc['mass_bal_et', 'f_et_v_ps_+']) * 100 / 36), '% increase from lysimeter scaling '
            'and ET averaging, respectively)')
print('fET←Ps v. PS: PW (', round((table1.loc['lys_scaled_et', 'f_et_v_ratio_+'] - table1.loc['mass_bal_et',
            'f_et_v_ratio_+']) * 100 / 36), '% and', round((table1.loc['avg_et', 'f_et_v_ratio_+'] - table1.loc['mass_bal_et',
            'f_et_v_ratio_+']) * 100 / 36), '%), fPS→ET v. PS (', round((table1.loc['lys_scaled_et', 'f_ps_v_ps_+'] -
            table1.loc['mass_bal_et', 'f_ps_v_ps_+']) * 100 / 36), '% and',
            round((table1.loc['avg_et', 'f_ps_v_ps_+'] - table1.loc['mass_bal_et', 'f_ps_v_ps_+']) * 100 / 36), '%), and ',
            'fPS→ET v. PS: PW (', round((table1.loc['lys_scaled_et', 'f_ps_v_ratio_+'] - table1.loc['mass_bal_et',
            'f_ps_v_ratio_+']) * 100 / 36), '% and', round((table1.loc['avg_et', 'f_ps_v_ratio_+'] - table1.loc['mass_bal_et',
            'f_ps_v_ratio_+']) * 100 / 36), '%)')
print('and increased the number of negative trends between fET←PS v. PW (',
            round((table1.loc['lys_scaled_et', 'f_et_v_pw_-'] - table1.loc['mass_bal_et', 'f_et_v_pw_-']) * 100 / 36),
            '% and', round((table1.loc['avg_et', 'f_et_v_pw_-'] - table1.loc['mass_bal_et', 'f_et_v_pw_-']) * 100 / 36),
            '%) and fPS→ET v. PW (', round((table1.loc['lys_scaled_et', 'f_ps_v_pw_-'] - table1.loc['mass_bal_et',
            'f_ps_v_pw_-']) * 100 / 36), '% and', round((table1.loc['avg_et', 'f_ps_v_pw_-'] - table1.loc['mass_bal_et',
            'f_ps_v_pw_-']) * 100 / 36), '%).')

lys_yr_list = evapotranspiration['Year'].tolist()
lys_et_in_df = []
ps_in_lys_et = []
pw_in_lys_et = []
ptot_in_lys_et = []
for i in range(len(df_lys['No Lag']['Year'])):
    if df_lys['No Lag']['Year'][i] in lys_yr_list and not np.isnan(evapotranspiration['annual_ET'][lys_yr_list.index(df_lys['No Lag']['Year'][i])]):
        lys_et_in_df.append(evapotranspiration['annual_ET'][lys_yr_list.index(df_lys['No Lag']['Year'][i])])
        ps_in_lys_et.append(df_lys['No Lag']['P_s'][i])
        pw_in_lys_et.append(df_lys['No Lag']['P_w'][i])
        ptot_in_lys_et.append(df_lys['No Lag']['Ptot'][i])


def plot_ols(x_vals, y_vals, xlabel, ylabel, title):
    x, y = zip(*sorted(zip(x_vals, y_vals)))
    meanx = sum(x) / len(x)
    X = sm.add_constant(x)
    res_ols = sm.OLS(y, X).fit()
    fitvals = res_ols.fittedvalues.tolist()
    slope_pval = res_ols.pvalues[1]
    if slope_pval < 0.1:
        x_val_ci = range(round(min(x_vals) - 5), round(max(x_vals) + 5), round((max(x_vals) - min(x_vals))/20))
        int_se, slope_se = res_ols.bse
        df = res_ols.df_resid
        t_crit = abs(scipy.stats.t.ppf(q=0.025, df=df))
        ci_upp = [0] * len(x_val_ci)
        ci_low = [0] * len(x_val_ci)
        for i in range(len(x_val_ci)):
            ci_upp[i] = res_ols.params[1] * x_val_ci[i] + res_ols.params[0] + t_crit * math.sqrt((abs(x_val_ci[i] -
                                                                            meanx) * slope_se) ** 2 + int_se ** 2)
            ci_low[i] = res_ols.params[1] * x_val_ci[i] + res_ols.params[0] - t_crit * math.sqrt((abs(x_val_ci[i] -
                                                                meanx) * slope_se) ** 2 + int_se ** 2)
        plt.plot(x_val_ci, ci_low, color='black')
        plt.plot(x_val_ci, ci_upp, color='black')
        plt.plot(x, fitvals)
    plt.title('\n'.join(
        wrap(title,
             60)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y)
    print(slope_pval)

plot_ols(df_lys["No Lag"]['Ptot'], df_lys["No Lag"]['AllP_del'], 'Annual Precipitation (mm)',
                                                                               'Annual Weighted Average of Precipitation δ$^{18}$O', 'Greater Amounts of Annual Precipitation Correlate with Lower Average Annual Precipitation Isotope Values')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\annualPdel.eps', dpi=400)
plt.show()

plot_ols(ps_in_lys_et, lys_et_in_df, 'Summer Precipitation (mm)', 'Lysimeter Evapotranspiration (mm)',
         'Summer Precipitation Amount Does Not Correlate with Evapotranspiration Calculated from Lysimeter Weights')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\summerP.eps', dpi=400)
plt.show()

plot_ols(pw_in_lys_et, lys_et_in_df, 'Winter Precipitation (mm)', 'Lysimeter Evapotranspiration (mm)',
         'Winter Precipitation Amount Does Not Correlate with Evapotranspiration Calculated from Lysimeter Weights')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\winterP.eps', dpi=400)
plt.show()

plot_ols(ptot_in_lys_et, lys_et_in_df, 'Annual Precipitation (mm)', 'Lysimeter Evapotranspiration (mm)',
         'Annual Precipitation Amount Does Not Correlate with Evapotranspiration Calculated from Lysimeter Weights')
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\AnnualP.eps', dpi=400)
plt.show()



