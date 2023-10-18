from pre_endsplit import preprocessing, sum_precip_totals
from plot_endsplit import diagram_in_out, multi_year_endsplit, plot_del_figure, calc_year_indices, plot_correlations, ols_slope_int
from bootstrap_endsplit import plot_panels_q, plot_panels_et
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import sem, stats
from textwrap import wrap
import statistics as stats
import math

# Import data
data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHB.csv')
daily_data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHBDaily.csv')

# Extract key columns
sampling_dates = data.loc[:, 'date']
precip_isotope = data.loc[:, 'Precip_18O_combined']
interval = data.loc[:, 'time_interval_d']
date_daily = daily_data['date'].to_list()
precip_mm = daily_data['Sum(Precip_mm)'].tolist()
temperature = daily_data['Mean(airtemp_C)'].tolist()


# EMS with All Rietholzbach discharge data
stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
q_all = daily_data['Summed_RHB_discharge_mm'].tolist()
fluxes_all, iso_data_all, pwt_all, qwt_all, years_all = preprocessing(sampling_dates, precip_isotope, interval,
        stream_isotope, date_daily, precip_mm, q_all, temperature)

# EMS with Upper Rietholzbach discharge data
stream_isotope_upp = data.loc[:, 'Upper_RHB_18O_combined']
fluxes_upp, iso_data_upp, pwt_upp, qwt_upp, years_upp = preprocessing(sampling_dates, precip_isotope, interval,
        stream_isotope_upp, date_daily, precip_mm, q_all, temperature)  # q_all is used because q_upper was unreliable


# Calculate summer and winter precipitation totals for each year of the upper catchment dataset as well as the
# fraction of precipitation falling as snow and the feb-apr avg temperature for figure S4
summer_precip_upp = [0] * len(fluxes_upp)
winter_precip_upp = [0] * len(fluxes_upp)
snow_upp = [0] * len(fluxes_upp)
f_precip_as_snow = [0] * len(fluxes_upp)
temp = [[] for _ in fluxes_upp]
for y in range(len(fluxes_upp)):
    for d in range(len(fluxes_upp[y]['P'])):
        if fluxes_upp[y]['Pcat'][d] == 'summer':
            summer_precip_upp[y] += fluxes_upp[y]['P'][d]
        if fluxes_upp[y]['Pcat'][d] == 'winter':
            winter_precip_upp[y] += fluxes_upp[y]['P'][d]
        if fluxes_upp[y]['Tcat'][d] == 'snow':
            snow_upp[y] += fluxes_upp[y]['P'][d]
        if int(fluxes_upp[y]['dates'][d].strftime('%m')) > 1 and int(fluxes_upp[y]['dates'][d].strftime('%m')) < 5:
            temp[y].append(fluxes_upp[y]['T'][d])
    f_precip_as_snow[y] = snow_upp[y] / (summer_precip_upp[y] + winter_precip_upp[y])

temp_feb_apr = [0] * len(fluxes_upp)
for i in range(len(temp_feb_apr)):
    temp_feb_apr[i] = stats.mean(temp[i])

# Plot figure S4: seasonal precipitation amounts by fraction falling as snow and feb-apr temperature
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
y_ticks = [0.05, 0.10, 0.15, 0.20]
axs[0, 0].set_title('A.                                        ', fontsize=14)
axs[0, 0].set_yticks(y_ticks)
axs[0, 0].set_ylabel('\n'.join(wrap('Fraction of Precipitation Falling as Snow (unitless)', 30)), fontsize=14)
ols_slope_int(summer_precip_upp, f_precip_as_snow, plot=axs[0, 0])

axs[0, 1].set_title('B.                                        ', fontsize=14)
axs[0, 1].set_yticks(y_ticks)
axs[0, 1].set_ylabel('\n'.join(wrap('Fraction of Precipitation Falling as Snow (unitless)', 30)), fontsize=14)
ols_slope_int(winter_precip_upp, f_precip_as_snow, plot=axs[0, 1])

axs[1, 0].set_title('C.                                        ', fontsize=14)
axs[1, 0].set_ylabel('\n'.join(wrap('Feb - Apr Mean Temperature (degrees Celsius)', 30)), fontsize=14)
axs[1, 0].set_xlabel('Summer Precipitaton (mm)', fontsize=14)
ols_slope_int(summer_precip_upp, temp_feb_apr, plot=axs[1, 0])

axs[1, 1].set_title('D.                                        ', fontsize=14)
axs[1, 1].set_ylabel('\n'.join(wrap('Feb - Apr Mean Temperature (degrees Celsius)', 30)), fontsize=14)
axs[1, 1].set_xlabel('Winter Precipitaton (mm)', fontsize=14)
ols_slope_int(winter_precip_upp, temp_feb_apr, plot=axs[1, 1])

plt.tight_layout()
#fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\fractionsnow.svg', dpi=500)
plt.show()


# EMS with lysimeter seepage data
isotope_lysimeter = data.loc[:, 'Lysimeter_18O_combined']
lysimeter_seepage = daily_data['lys_seep_mm/day'].tolist()
fluxes_lys, iso_data_lys, pwt_lys, qwt_lys, years_lys = preprocessing(sampling_dates, precip_isotope, interval,
                                            isotope_lysimeter, date_daily, precip_mm, lysimeter_seepage, temperature)


# Figure S3: Isotope Values
plot_del_figure(q_all, stream_isotope, sampling_dates, date_daily, stream_isotope_upp, lysimeter_seepage,
                    isotope_lysimeter, precip_mm, precip_isotope, interval, iso_data_all, qwt_all, iso_data_upp,
                    qwt_upp, iso_data_lys, qwt_lys)


# Figure 1: Longterm Split Diagrams
longterm_all, lt_table_all = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))
longterm_upp, lt_table_upp = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))
longterm_lys, lt_table_lys = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))


fig, axs = plt.subplots(1, 3, figsize=(25, 10))
diagram_in_out(axs[0], lt_table_all, longterm_all, 'All RHB', Pw_ET_amt=80)
diagram_in_out(axs[1], lt_table_upp, longterm_upp, 'Upper RHB', Pw_ET_amt=20)
diagram_in_out(axs[2], lt_table_lys, longterm_lys, 'Lysimeter', Pw_ET_amt=20)
fig.tight_layout()
#fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\LongtermSplitDiagrams.svg', dpi=500)
plt.show()

precip_df = sum_precip_totals(date_daily, precip_mm)


# Figure 2: Split Diagrams for Upper RHB
fig, axs = plt.subplots(2, 2, figsize=(17, 18))
year_indices = calc_year_indices(precip_df, years_upp, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[0, 0], table, df, 'Wetter Summers', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30,
               Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[0, 1], table, df, 'Drier Summers', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Wetter Winters', Pw_ET_pct=20, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Drier Winters', Pw_ET_amt=85, Ps_Qs_amt=-30)

fig.tight_layout()
fig.savefig(r'C:\Users\User\Documents\UNR\Summer2023\Rietholzbach\UpperRHBSplitDiagrams_Oct_to_Sep.svg', dpi=500)
plt.show()


# Figure S5: Split Diagrams for All RHB
fig, axs = plt.subplots(2, 2, figsize=(17, 17))
year_indices = calc_year_indices(precip_df, years_all, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 0], table, df, 'Wetter Summers', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30,
               Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_all, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 1], table, df, 'Drier Summers', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_all, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Wetter Winters', Pw_ET_pct=20, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_all, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Drier Winters', Ps_ET_pct=50, Pw_ET_amt=85, Pw_ET_pct=120,
               Ps_Qs_amt=-30, Ps_Qs_pct=-20)
fig.tight_layout()
#fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\AllRHBSplitDiagrams.svg', dpi=500)
plt.show()


# Figure S6: Split Diagrams for Lysimeter
fig, axs = plt.subplots(2, 2, figsize=(17, 17))

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[0, 0], table, df, 'Wetter Summers', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30,
               Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[0, 1], table, df, 'Drier Summers', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Wetter Winters', Pw_ET_pct=20, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Drier Winters', Ps_ET_pct=50, Pw_ET_amt=85, Pw_ET_pct=120,
               Ps_Qs_amt=-30, Ps_Qs_pct=-20)
fig.tight_layout()
#fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\LysSplitDiagrams.svg', dpi=500)
plt.show()


# Figure 3 Bootstrapping Results for Q
X_axis = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660,
          670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880,
          890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]
df_upp_q, summer_results_q_upp, winter_results_q_upp = plot_panels_q(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, X_axis,
                                                                    'Upper RHB')



# Figure 4 Bootstrapping Results for ET
evapotranspiration = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\Evapotranspiration_Hirschi.csv')
df_upp_et, summer_results_et_upp, winter_results_et_upp = plot_panels_et(years_upp, iso_data_upp, fluxes_upp, pwt_upp,
                                                                    qwt_upp, evapotranspiration, X_axis, 'Upper RHB')

df_annual_summ = df_upp_q.loc[:, ('Year', 'Ptot', 'P_s', 'P_w', 'Pdel_s', 'Pdel_w', 'Q', 'Q_s', 'Q_w', 'Qdel', 'Qdel_s', 'Qdel_w', 'ET')]
df_annual_summ = df_annual_summ.rename(columns={'Q': 'Q_i', 'ET': 'ET_i'})
df_annual_summ = df_annual_summ.join(df_upp_et['Q'])
df_annual_summ = df_annual_summ.join(df_upp_et['ET'])
df_annual_summ = df_annual_summ.rename(columns={'Q': 'Q_adj', 'ET': 'ET_adj'})
#df_annual_summ.to_csv(r'C:\Users\User\Documents\UNR\Summer2023\Rietholzbach\df_upp_annual_summary.csv')

# Figures S7 and S8
plot_correlations(df_upp_et, 'P_s', 'Upper RHB')
plot_correlations(df_upp_et, 'P_w', 'Upper RHB')


# Figures S9, S10, S13, S14: Plot bootstrapping results and variable correlations for All RHB
df_all_q, summer_results_q_all, winter_results_q_all = plot_panels_q(iso_data_all, fluxes_all, pwt_all, qwt_all,
                                                                X_axis, 'All RHB')
df_all_et, summer_results_et_all, winter_results_et_all = plot_panels_et(years_all, iso_data_all, fluxes_all, pwt_all,
                                                                qwt_all, evapotranspiration, X_axis, 'All RHB')
plot_correlations(df_all_et, 'P_s', 'All RHB')
plot_correlations(df_all_et, 'P_w', 'All RHB')


# Figures S11, S12, S15, S16: Plot bootstrapping results and variable correlations for Lysimeter
df_lys_q, summer_results_q_lys, winter_results_q_lys = plot_panels_q(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys,
                                                                X_axis, 'Lysimeter')
df_lys_et, summer_results_et_lys, winter_results_et_lys = plot_panels_et(years_lys, iso_data_lys, fluxes_lys, pwt_lys,
                                                                qwt_lys, evapotranspiration, X_axis, 'Lysimeter')
plot_correlations(df_lys_et, 'P_s', 'Lysimeter')
plot_correlations(df_lys_et, 'P_w', 'Lysimeter')


# Figure S1: Annual ET Amounts for each dataset
x = [1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
ET_all_upp = [0] * len(x)
NP_all = [0] * len(x)
NP_upp = [0] * len(x)
ET_lys = [0] * len(x)
NP_lys = [0] * len(x)
ET_wts = [0] * len(x)

for i in range(len(x)):
    if x[i] in df_upp_q['Year'].tolist():
        ET_all_upp[i] = df_upp_q['ET'][df_upp_q['Year'].tolist().index(x[i])]
    if x[i] in df_all_et['Year'].tolist():
        NP_all[i] = df_all_et['ET'][df_all_et['Year'].tolist().index(x[i])]
    if x[i] in df_upp_et['Year'].tolist():
        NP_upp[i] = df_upp_et['ET'][df_upp_et['Year'].tolist().index(x[i])]
    if x[i] in df_lys_q['Year'].tolist():
        ET_lys[i] = df_lys_q['ET'][df_lys_q['Year'].tolist().index(x[i])]
    if x[i] in df_lys_et['Year'].tolist():
        NP_lys[i] = df_lys_et['ET'][df_lys_et['Year'].tolist().index(x[i])]
    if x[i] in evapotranspiration['Year'].tolist():
        ET_wts[i] = evapotranspiration['annual_ET'][evapotranspiration['Year'].tolist().index(x[i])]

plt.figure(figsize=(7, 5))
plt.plot(x, ET_all_upp, color='purple', label='All and Upper RHB Water Balance')
plt.plot(x[0:15], NP_all[0:15], color='blue', label='All RHB Net Percolation')
plt.plot(x, NP_upp, color='cyan', label='Upper RHB Net Percolation')
plt.plot(x, ET_lys, color='r', label='Lysimeter Water Balance')
plt.plot(x, NP_lys, color='orange', label='Lysimeter Net Percolation')
plt.plot(x, ET_wts, color='black', label='Lysimeter Mass Variations')
for i in [ET_all_upp, NP_all, NP_all[0:15], NP_upp, ET_lys, NP_lys, ET_wts]:
    print(sem([x for x in i if x != 0]))
plt.xlabel('Year')
plt.xticks(ticks=[1995, 2000, 2005, 2010])
plt.ylabel('Evapotranspiration (mm)')
plt.legend(labelspacing=0.1)
#plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\AnnualETAmountsLines.svg', dpi=500)
plt.show()


# Redo Figure 1 Longterm Split Diagrams Given Undercatch Corrections for Rainfall
longterm_all_orig, lt_table_all_orig = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))
longterm_upp_orig, lt_table_upp_orig = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))
longterm_lys_orig, lt_table_lys_orig = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))

precip = []
snow = []
for i in range(len(fluxes_upp)):
    for d in range(len(fluxes_upp[i]['P'])):
        precip.append(fluxes_upp[i]['P'][d])
        if fluxes_upp[i]['Tcat'][d] == 'snow':
            snow.append(fluxes_upp[i]['P'][d])
print(sum(snow) / sum(precip))
fluxes_all, iso_data_all, pwt_all, qwt_all, years_all = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope, date_daily, precip_mm, q_all, temperature, undercatch_type='rain')
fluxes_upp, iso_data_upp, pwt_upp, qwt_upp, years_upp = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope_upp, date_daily, precip_mm, q_all, temperature, undercatch_type='rain')
fluxes_lys, iso_data_lys, pwt_lys, qwt_lys, years_lys = preprocessing(sampling_dates, precip_isotope, interval,
                                            isotope_lysimeter, date_daily, precip_mm, lysimeter_seepage, temperature, undercatch_type='rain')
longterm_all, lt_table_all = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))
longterm_upp, lt_table_upp = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))
longterm_lys, lt_table_lys = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))
print('rain adjusted difference from original')
print('All RHB f_ET', lt_table_all['f.summer']['ET'] - lt_table_all_orig['f.summer']['ET'])
print('Upper RHB f_ET', lt_table_upp['f.summer']['ET'] - lt_table_upp_orig['f.summer']['ET'])
print('Lysimeter RHB f_ET', lt_table_lys['f.summer']['ET'] - lt_table_lys_orig['f.summer']['ET'])
print('All RHB f_Ps', lt_table_all['eta.summer']['ET'] - lt_table_all_orig['eta.summer']['ET'])
print('Upper RHB f_Ps', lt_table_upp['eta.summer']['ET'] - lt_table_upp_orig['eta.summer']['ET'])
print('Lysimeter RHB f_Ps', lt_table_lys['eta.summer']['ET'] - lt_table_lys_orig['eta.summer']['ET'])


# Redo Figure 1 Longterm Split Diagrams Given Undercatch Corrections for Snowfall
fluxes_all, iso_data_all, pwt_all, qwt_all, years_all = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope, date_daily, precip_mm, q_all, temperature, undercatch_type='snow')
fluxes_upp, iso_data_upp, pwt_upp, qwt_upp, years_upp = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope_upp, date_daily, precip_mm, q_all, temperature, undercatch_type='snow')
fluxes_lys, iso_data_lys, pwt_lys, qwt_lys, years_lys = preprocessing(sampling_dates, precip_isotope, interval,
                                            isotope_lysimeter, date_daily, precip_mm, lysimeter_seepage, temperature, undercatch_type='snow')
longterm_all, lt_table_all = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))
longterm_upp, lt_table_upp = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))
longterm_lys, lt_table_lys = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))
print('snow adjusted difference from original')
print('All RHB f_ET', lt_table_all['f.summer']['ET'] - lt_table_all_orig['f.summer']['ET'])
print('Upper RHB f_ET', lt_table_upp['f.summer']['ET'] - lt_table_upp_orig['f.summer']['ET'])
print('Lysimeter RHB f_ET', lt_table_lys['f.summer']['ET'] - lt_table_lys_orig['f.summer']['ET'])
print('All RHB f_Ps', lt_table_all['eta.summer']['ET'] - lt_table_all_orig['eta.summer']['ET'])
print('Upper RHB f_Ps', lt_table_upp['eta.summer']['ET'] - lt_table_upp_orig['eta.summer']['ET'])
print('Lysimeter RHB f_Ps', lt_table_lys['eta.summer']['ET'] - lt_table_lys_orig['eta.summer']['ET'])


# Redo Figure 1 Longterm Split Diagrams Given Undercatch Corrections for Both Rainfall and Snowfall
fluxes_all, iso_data_all, pwt_all, qwt_all, years_all = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope, date_daily, precip_mm, q_all, temperature, undercatch_type='both')
fluxes_upp, iso_data_upp, pwt_upp, qwt_upp, years_upp = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope_upp, date_daily, precip_mm, q_all, temperature, undercatch_type='both')
fluxes_lys, iso_data_lys, pwt_lys, qwt_lys, years_lys = preprocessing(sampling_dates, precip_isotope, interval,
                                            isotope_lysimeter, date_daily, precip_mm, lysimeter_seepage, temperature, undercatch_type='both')
longterm_all, lt_table_all = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))
longterm_upp, lt_table_upp = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))
longterm_lys, lt_table_lys = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))
print('both adjusted difference from original')
print('All RHB f_ET', lt_table_all['f.summer']['ET'] - lt_table_all_orig['f.summer']['ET'])
print('Upper RHB f_ET', lt_table_upp['f.summer']['ET'] - lt_table_upp_orig['f.summer']['ET'])
print('Lysimeter RHB f_ET', lt_table_lys['f.summer']['ET'] - lt_table_lys_orig['f.summer']['ET'])
print('All RHB f_Ps', lt_table_all['eta.summer']['ET'] - lt_table_all_orig['eta.summer']['ET'])
print('Upper RHB f_Ps', lt_table_upp['eta.summer']['ET'] - lt_table_upp_orig['eta.summer']['ET'])
print('Lysimeter RHB f_Ps', lt_table_lys['eta.summer']['ET'] - lt_table_lys_orig['eta.summer']['ET'])

