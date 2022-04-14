import pandas as pd
import math
import statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from PdelFigure import CalcPrecip, CalcQ, PlotPdelFigure
from preprocessing import ConvertDateTime, ListYears, FindStartOfSampling, ListUnusableDates, SplitFluxesByHydrologicYear, SplitIsotopesByHydrologicYear
from cleaning import SumPrecipitationAndRunoff, RemoveNanSamples, IdentifyMissingData
from calculations import LysimeterUndercatch, EndSplit
from plot import PlotFraction, PlotAmount, PlotODR, LysCorrectedEndsplit, line
from sklearn.linear_model import LinearRegression
import scipy as scipy
import scipy.odr as odr
from textwrap import wrap

# Read data
data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHB.csv')
daily_data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHBDaily.csv')

# Isotope data variables
Sampling_dates = data.loc[:, 'date']
Precip_isotope = data.loc[:, 'Precip_18O_combined']
Interval = data.loc[:, 'time_interval_d']

# Flux data variables
date_daily = daily_data['date'].to_list()
P = daily_data['Sum(Precip_mm)'].tolist()
temperature = daily_data['Mean(airtemp_C)'].tolist()

plots = {'All':[], 'Upper':[], 'Lysimeter':[]}

conf_int = {'All':[], 'Upper':[], 'Lysimeter':[]}

wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month = CalcPrecip(Sampling_dates, P, Precip_isotope, Interval, 4, 9)

def WorkFlowEndSplit(dates, Precip_isotope, Interval, Stream_isotope, dates_daily, P, Q, Watershed, temperature):
    Sampling_dates = ConvertDateTime(dates)

    date_daily = ConvertDateTime(dates_daily)

    years = ListYears(Sampling_dates)

    years_daily = ListYears(date_daily)

    first_month, first_year = FindStartOfSampling(Sampling_dates, date_daily, Precip_isotope, P, Stream_isotope, Q,
                                                  side='start')

    last_month, last_year = FindStartOfSampling(Sampling_dates[::-1], date_daily[::-1], Precip_isotope.values[::-1],
                                                P[::-1], Stream_isotope.values[::-1], Q[::-1], side='end')

    no_count_date = ListUnusableDates(Sampling_dates, first_month, years.index(int(first_year)), last_month,
                                      years.index(int(last_year)), years)

    no_count_date_daily = ListUnusableDates(date_daily, first_month, years_daily.index(int(first_year)), last_month,
                                            years_daily.index(int(last_year)), years_daily)

    fluxes = SplitFluxesByHydrologicYear(date_daily, years_daily,no_count_date_daily, P, Q, temperature)

    date_by_year, precip_d_year, Pdelcat_with_nan, runoff_d_year, Qdelcat_with_nan, interval_by_year = \
         SplitIsotopesByHydrologicYear(Sampling_dates, Interval, years, no_count_date, Precip_isotope, Stream_isotope, first_year)

    #print('Missing isotope data')
    # The following line shows missing isotope data if relevant lines in IdentifyMissingData are uncommented
    #remove_years = IdentifyMissingData(fluxes, interval_by_year)

    interval = [[] for _ in fluxes]
    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            interval[i].append(1)

    remove_years_daily = IdentifyMissingData(fluxes, interval)
    if remove_years_daily:
        print(' ')
        print('Remove years', remove_years_daily, 'due to missing flux data')

    Pwt, Qwt = SumPrecipitationAndRunoff(date_by_year, fluxes, precip_d_year, runoff_d_year)

    Qdel, Qdelcat = RemoveNanSamples(years, runoff_d_year, Qdelcat_with_nan)

    Pdel, Pdelcat = RemoveNanSamples(years, precip_d_year, Pdelcat_with_nan)

    # Analysis using values from year of interest (assumes no lag or mixing)
    df_no_lag = pd.DataFrame(columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                                          'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], fluxes[i]['P'],
                                       fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=df_no_lag.columns)
            df_no_lag = df_no_lag.append(new_row, ignore_index=True)

    no_lag_P, no_lag_plot, no_lag_x_ci, no_lag_ci_low, no_lag_ci_high = PlotFraction(df_no_lag['P_s'], df_no_lag['P_s_se'], df_no_lag['f_ET'], df_no_lag['f_ET_se'], "No Lag", Watershed)

    # PlotAmount function plots summer precipitation by the total amount of evapotranspiration
    PlotAmount(df_no_lag['P_s'], df_no_lag['P_s_se'], df_no_lag['ET'], df_no_lag['ET_se'], "No Lag", Watershed)

    # Lagged flow analysis, 1 year
    # Repeat end-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest
    df_lag_1 = pd.DataFrame(columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                                      'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    lag_years = 1


    for i in range(lag_years, len(years)):
        if Pwt[i-lag_years] and fluxes[i]['Pcat'] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i-lag_years], Qdel[i], Pwt[i-lag_years], Qwt[i], Pdelcat[i-lag_years], Qdelcat[i],
                               fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=df_lag_1.columns)
            df_lag_1 = df_lag_1.append(new_row, ignore_index=True)

    lag_1_P, lag_1_plot, lag_1_x_ci, lag_1_ci_low, lag_1_ci_high = PlotFraction(df_lag_1['P_s'], df_lag_1['P_s_se'], df_lag_1['f_ET'], df_lag_1['f_ET_se'], "Lagged One Year", Watershed)

    # Lagged flow analysis, 2 years
    # Repeat end-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest

    df_lag_2 = pd.DataFrame(columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                                     'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    lag_years = 2

    for i in range(lag_years, len(years)):
        if Pwt[i-lag_years] and fluxes[i]['Pcat'] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i - lag_years], Qdel[i], Pwt[i - lag_years], Qwt[i], Pdelcat[i - lag_years],
                               Qdelcat[i], fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=df_lag_2.columns)
            df_lag_2 = df_lag_2.append(new_row, ignore_index=True)

    lag_2_P, lag_2_plot, lag_2_x_ci, lag_2_ci_low, lag_2_ci_high = PlotFraction(df_lag_2['P_s'], df_lag_2['P_s_se'], df_lag_2['f_ET'], df_lag_2['f_ET_se'], "Lagged Two Years",
                 Watershed)

    # Mixed groundwater analysis
    # Repeat end-member splitting, but keep constant values of Pdel_bar each season for all years using the summer
    # and winter isotope values averaged over the entire study period

    isotope_vals = []
    weights = []
    season = []

    for i in range(len(Pwt)):
        for d in range(len(Pwt[i])):
            isotope_vals.append(Pdel[i][d])
            weights.append(Pwt[i][d])
            if Pdelcat[i][d] == "winter":
                season.append("winter")
            else:
                season.append("summer")

    df_mixed = pd.DataFrame(columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                                     'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            results = EndSplit(isotope_vals, Qdel[i], weights, Qwt[i], season, Qdelcat[i], fluxes[i]['P'],
                               fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=df_mixed.columns)
            df_mixed = df_mixed.append(new_row, ignore_index=True)

    mixed_P, mixed_plot, mixed_x_ci, mixed_ci_low, mixed_ci_high = PlotFraction(df_mixed['P_s'], df_mixed['P_s_se'], df_mixed['f_ET'], df_mixed['f_ET_se'], "Mixed", Watershed)

    plots[Watershed].extend([no_lag_P, no_lag_plot, lag_1_P, lag_1_plot, lag_2_P, lag_2_plot, mixed_P, mixed_plot])

    conf_int[Watershed].extend([no_lag_x_ci, no_lag_ci_low, no_lag_ci_high, lag_1_x_ci, lag_1_ci_low, lag_1_ci_high, lag_2_x_ci, lag_2_ci_low, lag_2_ci_high, mixed_x_ci, mixed_ci_low, mixed_ci_high])

    # Adjust both rain and snowfall for undercatch

    precip_adjusted_fluxes = LysimeterUndercatch(fluxes, category='both')

    Pwt, Qwt = SumPrecipitationAndRunoff(date_by_year, precip_adjusted_fluxes, precip_d_year, runoff_d_year)

    undercatch_both = pd.DataFrame(columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                                 'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], precip_adjusted_fluxes[i]['P'],
                           precip_adjusted_fluxes[i]['Q'], precip_adjusted_fluxes[i]['Pcat'], precip_adjusted_fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=undercatch_both.columns)
            undercatch_both = undercatch_both.append(new_row, ignore_index=True)

    # Adjust only rainfall for undercatch

    rain_adjusted_fluxes = LysimeterUndercatch(fluxes, category='rain')

    Pwt, Qwt = SumPrecipitationAndRunoff(date_by_year, rain_adjusted_fluxes, precip_d_year, runoff_d_year)

    undercatch_rain = pd.DataFrame(
        columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                 'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i],
                               rain_adjusted_fluxes[i]['P'],
                               rain_adjusted_fluxes[i]['Q'], rain_adjusted_fluxes[i]['Pcat'],
                               rain_adjusted_fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=undercatch_rain.columns)
            undercatch_rain = undercatch_rain.append(new_row, ignore_index=True)

    # Adjust only snowfall for undercatch

    snow_adjusted_fluxes = LysimeterUndercatch(fluxes, category='snow')

    Pwt, Qwt = SumPrecipitationAndRunoff(date_by_year, snow_adjusted_fluxes, precip_d_year, runoff_d_year)

    undercatch_snow = pd.DataFrame(
        columns=['Year', 'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET',
                 'f_ET_se', 'ET', 'ET_se', 'AllP_del'])

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            results = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i],
                               snow_adjusted_fluxes[i]['P'],
                               snow_adjusted_fluxes[i]['Q'], snow_adjusted_fluxes[i]['Pcat'],
                               snow_adjusted_fluxes[i]['Qcat'])
            results[0] = years[i]
            new_row = pd.Series(results, index=undercatch_snow.columns)
            undercatch_snow = undercatch_snow.append(new_row, ignore_index=True)

    difference_both = []
    difference_rain = []
    difference_snow = []
    weighted_mean_diff_both = []
    weighted_mean_diff_rain = []
    weighted_mean_diff_snow = []
    weights_both = []
    weights_rain = []
    weights_snow = []

    for i in range(len(df_no_lag['f_ET'])):
        difference_both.append(df_no_lag['f_ET'][i] - undercatch_both['f_ET'][i])
        difference_rain.append(df_no_lag['f_ET'][i] - undercatch_rain['f_ET'][i])
        difference_snow.append(df_no_lag['f_ET'][i] - undercatch_snow['f_ET'][i])
        weighted_mean_diff_both.append(difference_both[i] * (1 / undercatch_both['f_ET_se'][i]))
        weights_both.append(1 / undercatch_both['f_ET_se'][i])
        weighted_mean_diff_rain.append(difference_rain[i] * (1 / undercatch_rain['f_ET_se'][i]))
        weights_rain.append(1 / undercatch_rain['f_ET_se'][i])
        weighted_mean_diff_snow.append(difference_snow[i] * (1 / undercatch_snow['f_ET_se'][i]))
        weights_snow.append(1 / undercatch_snow['f_ET_se'][i])

    print('')
    print('Undercatch Sensitivity Analysis for ', Watershed)
    print("Adjusted rainfall mean difference: ", stats.mean(difference_rain),
          ' inverse error weighted: ', sum(weighted_mean_diff_rain)/sum(weights_rain))
    print('Adjusted snow mean difference: ', stats.mean(difference_snow),
          ' inverse error weighted: ', sum(weighted_mean_diff_snow)/sum(weights_snow))
    print('Adjusted rain and snow mean difference: ', stats.mean(difference_both),
          ' inverse error weighted: ', sum(weighted_mean_diff_both)/sum(weights_both))

    return df_no_lag, df_lag_1, df_lag_2, df_mixed

# All Rietholzbach Catchment

Stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
Q = daily_data['Summed_RHB_discharge_mm'].tolist()
df_no_lag_all, df_lag_1_all, df_lag_2_all, df_mixed_all = WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Stream_isotope, date_daily, P, Q, "All", temperature)
wtd_mean_stream_all, error_stream_all = CalcQ(Q, Stream_isotope, Sampling_dates, date_daily)

# Upper Rietholzbach

Stream_isotope_upper = data.loc[:, 'Upper_RHB_18O_combined']
Q_upper = daily_data['SummedDischarge(upper_RHB_mm/h)'].tolist()

df_no_lag_upper, df_lag_1_upper, df_lag_2_upper, df_mixed_upper = WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Stream_isotope_upper, date_daily, P, Q_upper, "Upper", temperature)

wtd_mean_stream_upper, error_stream_upper = CalcQ(Q_upper, Stream_isotope_upper, Sampling_dates, date_daily)

# Lysimeter

Isotope_lysimeter_seepage = data.loc[:, 'Lysimeter_18O_combined']
Amount_lysimeter_seepage = daily_data['lys_seep_mm/day'].tolist()

df_no_lag_lys, df_lag_1_lys, df_lag_2_lys, df_mixed_lys = WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Isotope_lysimeter_seepage, date_daily, P, Amount_lysimeter_seepage, "Lysimeter", temperature)

wtd_mean_stream_lys, error_stream_lys = CalcQ(Amount_lysimeter_seepage, Isotope_lysimeter_seepage, Sampling_dates, date_daily)

wtd_mean_stream = [wtd_mean_stream_all, wtd_mean_stream_upper, wtd_mean_stream_lys]
error_stream = [error_stream_all, error_stream_upper, error_stream_lys]
stream_label = ["All", "Upper", "Lys"]
colors = ["blue", "orange", "red"]

PlotPdelFigure(4, 9, wtd_mean_summer, s_error_summer, wtd_mean_winter, s_error_winter, wtd_mean_per_month, s_error_per_month, wtd_mean_stream, error_stream, stream_label, colors)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Inverse Error Weighted Regression, Significant (p<0.1) Slopes')
axs[0, 0].set(ylabel='Fraction of ET from Summer')
axs[1, 0].set(xlabel='Summer Precipitation (mm)', ylabel='Fraction of ET from Summer')
axs[1, 1].set(xlabel='Summer Precipitation (mm)')
axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
methods = ['No Lag', 'Lagged 1 Year', 'Lagged 2 Years', 'Mixed']
indices = [[0, 1, 0, 1, 2], [2, 3, 3, 4, 5], [4, 5, 6, 7, 8], [6, 7, 9, 10, 11]]
for i in range(len(axes)):
    axes[i].plot(plots['All'][indices[i][0]], plots['All'][indices[i][1]], color='blue', linewidth=3, label='All')
    axes[i].plot(conf_int['All'][indices[i][2]], conf_int['All'][indices[i][3]], color='blue')
    axes[i].plot(conf_int['All'][indices[i][2]], conf_int['All'][indices[i][4]], color='blue')
    axes[i].plot(plots['Upper'][indices[i][0]], plots['Upper'][indices[i][1]], color='orange', linewidth=3, label='Upper')
    axes[i].plot(conf_int['Upper'][indices[i][2]], conf_int['Upper'][indices[i][3]], color='orange')
    axes[i].plot(conf_int['Upper'][indices[i][2]], conf_int['Upper'][indices[i][4]], color='orange')
    axes[i].plot(plots['Lysimeter'][indices[i][0]], plots['Lysimeter'][indices[i][1]], color='green', linewidth=3, label='Lysimeter')
    axes[i].plot(conf_int['Lysimeter'][indices[i][2]], conf_int['Lysimeter'][indices[i][3]], color='green')
    axes[i].plot(conf_int['Lysimeter'][indices[i][2]], conf_int['Lysimeter'][indices[i][4]], color='green')
    axes[i].set_title(methods[i])
plt.legend(bbox_to_anchor = (1.05, 0.6))
fig.tight_layout(pad=1)
plt.show()

x_no_lag, reg_no_lag = PlotODR(df_no_lag_lys, df_no_lag_all, "All No Lag")

x_lag_1, reg_lag_1 = PlotODR(df_lag_1_lys, df_lag_1_all, "All Lagged 1 Year")

x_lag_2, reg_lag_2 = PlotODR(df_lag_2_lys, df_lag_2_all, "All Lagged 2 Years")

x_mixed, reg_mixed = PlotODR(df_mixed_lys, df_mixed_all, "All Mixed")

x_no_lag_upper, reg_no_lag_upper = PlotODR(df_no_lag_lys, df_no_lag_upper, "Upper No Lag")

x_lag_1_upper, reg_lag_1_upper = PlotODR(df_lag_1_lys, df_lag_1_upper, "Upper Lagged 1 Year")

x_lag_2_upper, reg_lag_2_upper = PlotODR(df_lag_2_lys, df_lag_2_upper, "Upper Lagged 2 Years")

x_mixed_upper, reg_mixed_upper = PlotODR(df_mixed_lys, df_mixed_upper, "Upper Mixed")

fig, axs = plt.subplots(1, 2)
labels = ['Fraction of ET from Summer Precipitation using Lysimeter','Fraction of ET from Summer Precipitation using All-Catchment Streamflow', 'Fraction of ET from Summer using Upper-Catchment Streamflow']
wrapped_labels = [ '\n'.join(wrap(l, 35)) for l in labels ]
axs[0].plot(x_no_lag, line(x_no_lag, reg_no_lag), label='No Lag', color='blue')
axs[0].plot(x_lag_1, line(x_lag_1, reg_lag_1), label='Lag 1 Year', color='red')
axs[0].plot(x_lag_2, line(x_lag_2, reg_lag_2), label='Lag 2 Years', color='purple')
axs[0].plot(x_mixed, line(x_mixed, reg_mixed), label='Mixed', color='orange')
axs[0].plot([0,1],[0,1], color='black', label='1:1')
axs[0].set_aspect('equal', adjustable='box')
axs[0].set(xlabel=wrapped_labels[0], ylabel=wrapped_labels[1], xlim=[0,1], ylim=[0,1])
axs[1].plot(x_no_lag_upper, line(x_no_lag_upper, reg_no_lag_upper), label='No Lag', color='blue')
axs[1].plot(x_lag_1_upper, line(x_lag_1_upper, reg_lag_1_upper), label='Lag 1 Year', color='red')
axs[1].plot(x_lag_2_upper, line(x_lag_2_upper, reg_lag_2_upper), label='Lag 2 Years', color='purple')
axs[1].plot(x_mixed_upper, line(x_mixed_upper, reg_mixed_upper), label='Mixed', color='orange')
axs[1].set(xlabel=wrapped_labels[0], ylabel=wrapped_labels[2], xlim=[0,1], ylim=[0,1])
axs[1].plot([0,1],[0,1], color='black', label='1:1')
axs[1].set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.suptitle('Inverse Error Weighted Orthogonal Distance Regression')
plt.legend()
plt.show()

evapotranspiration = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\Evapotranspiration.csv')

ET_lys_wts = [x for x in evapotranspiration['annual_ET'].tolist() if not math.isnan(x)]
avg_ET = sum(ET_lys_wts)/len(ET_lys_wts)
avg_ET_1990 = sum(ET_lys_wts[-16:])/len(ET_lys_wts[-16:])

plt.title('Annual Evapotranspiration from Lysimeter Weights')
plt.xlabel('Year')
plt.ylabel('Evapotranspiration (mm)')
plt.scatter(evapotranspiration['Year'], evapotranspiration['annual_ET'], label='Annual ET')
plt.plot(evapotranspiration['Year'], evapotranspiration['annual_ET'])
plt.plot([evapotranspiration['Year'][1], evapotranspiration['Year'][len(ET_lys_wts)]], [avg_ET, avg_ET], label='Average 1976-2007')
plt.plot([1990, evapotranspiration['Year'][len(ET_lys_wts)]], [avg_ET_1990, avg_ET_1990], label='Average 1990-2007')
plt.legend()
plt.show()

x = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]

ET_all = [0] * len(x)
ET_upper = [0] * len(x)
ET_lys = [0] * len(x)
ET_wts = [0] * len(x)
years_upper = x.copy()
years_lys = x.copy()
years_wts = x.copy()

for i in range(len(x)):
    years_upper[i] += 0.2
    years_lys[i] += 0.4
    years_wts[i] += 0.6
    if float(x[i]) in df_no_lag_all['Year'].tolist():
        ET_all[i] = df_no_lag_all['ET'][df_no_lag_all['Year'].tolist().index(x[i])]
    if x[i] in df_no_lag_upper['Year'].tolist():
        ET_upper[i] = df_no_lag_upper['ET'][df_no_lag_upper['Year'].tolist().index(x[i])]
    if x[i] in df_no_lag_lys['Year'].tolist():
        ET_lys[i] = df_no_lag_lys['ET'][df_no_lag_lys['Year'].tolist().index(x[i])]
    if x[i] in evapotranspiration['Year'].tolist():
        ET_wts[i] = evapotranspiration['annual_ET'][evapotranspiration['Year'].tolist().index(x[i])]

plt.title('Annual Evapotranspiration Values')
plt.xlabel('Year')
plt.ylabel('Evapotranspiration (mm)')

data = [ET_all, ET_upper, ET_lys, ET_wts]
plt.bar(x, data[0], color='purple', width=0.2, label='Rietholzbach Water Balance')
plt.bar(years_upper, data[1], color='b', width=0.2, label='Upper Rietholzbach Water Balance')
plt.bar(years_lys, data[2], color='r', width=0.2, label='Lysimeter Water Balance')
plt.bar(years_wts, data[3], color='black', width=0.2, label='Lysimeter Weights')

plt.legend()
plt.show()

undercatch = {'Year':[], 'Undercatch (mm)':[]}
for i in range(len(x)):
    if data[2][i] != 0 and data[3][i] != 0 and not np.isnan(data[2][i]) and not np.isnan(data[3][i]):
        undercatch['Year'].append(x[i])
        undercatch['Undercatch (mm)'].append(data[3][i] - data[2][i])
df_undercatch = pd.DataFrame(undercatch)

plt.title('Annual Precipitation Undercatch from Lysimeter Weights v. Water Balance')
plt.xlabel('Year')
plt.ylabel('Undercatch (mm)')
plt.plot(df_undercatch['Year'], df_undercatch['Undercatch (mm)'])
plt.scatter(df_undercatch['Year'], df_undercatch['Undercatch (mm)'])
plt.show()

# The folowing function corrects ET values by calculating Q = P - avg_ET, then
# Changes Ptot by adding precipitation undercatch for years in which ET from lysimeter weights is known
# ET_del is recalculated, then end-member mixing is used to calculate f_ET_summer
# Some of the output plots are wack, f_ET is way too low
# Should I use average undercatch instead?

f_ET_all_no_lag = LysCorrectedEndsplit(df_no_lag_all, avg_ET, "No Lag", "All", undercatch)

# Plots stopped displaying here, need to remove prior plt.show() to see the following plots
# Ask Scott about removing errors for f_ET_se > 10 for better visibility/viewability

f_ET_all_lag_1 = LysCorrectedEndsplit(df_lag_1_all, avg_ET, "Lagged One Year", "All", undercatch)

f_ET_all_lag_2 = LysCorrectedEndsplit(df_lag_2_all, avg_ET, "Lagged Two Years", "All", undercatch)

f_ET_all_mix = LysCorrectedEndsplit(df_mixed_all, avg_ET, "Mixed", "All", undercatch)

f_ET_upper_no_lag = LysCorrectedEndsplit(df_no_lag_upper, avg_ET, "No Lag", "Upper", undercatch)

f_ET_upper_lag_1 = LysCorrectedEndsplit(df_lag_1_upper, avg_ET, "Lagged One Year", "Upper", undercatch)

f_ET_upper_lag_2 = LysCorrectedEndsplit(df_lag_2_upper, avg_ET, "Lagged Two Years", "Upper", undercatch)

f_ET_upper_mix = LysCorrectedEndsplit(df_mixed_upper, avg_ET, "Mixed", "Upper", undercatch)

fig, axs = plt.subplots(3)
plt.suptitle("Distribution of Fraction of ET from Summer, Rietholzbach")

data_all = [df_no_lag_all['f_ET'], df_lag_1_all['f_ET'], df_lag_2_all['f_ET'], df_mixed_all['f_ET']]
data_upper = [df_no_lag_upper['f_ET'], df_lag_1_upper['f_ET'], df_lag_2_upper['f_ET'], df_mixed_upper['f_ET']]
data_lys = [df_no_lag_lys['f_ET'], df_lag_1_lys['f_ET'], df_lag_2_lys['f_ET'], df_mixed_lys['f_ET']]


axs[0].boxplot(data_all, showfliers=False, labels=[' ', ' ', ' ', ' '])
axs[1].boxplot(data_upper, showfliers=False, labels=[' ', ' ', ' ', ' '])
axs[2].boxplot(data_lys, showfliers=False, labels=['No Lag', 'Lag 1 Year', 'Lag 2 Years', 'Mixed'])
axs[0].set(ylabel='All')
axs[0].set_ylim(-1.5, 4)
axs[1].set(ylabel='Upper')
axs[1].set_ylim(-1.5, 4)
axs[2].set(ylabel='Lysimeter')
axs[2].set_ylim(-1.5, 4)
plt.show()
#plt.savefig('RHBboxplots.png')



