from pre_endsplit import preprocessing, convert_datetime
from plot_endsplit import diagram_in_out, multi_year_endsplit, plot_del_figure, calc_year_indices, plot_correlations
from bootstrap_endsplit import plot_panels_q, plot_panels_et
import matplotlib.pyplot as plt
import pandas as pd

# Import data
data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHB.csv')
daily_data = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\RHBDaily.csv')
sampling_dates = data.loc[:, 'date']
precip_isotope = data.loc[:, 'Precip_18O_combined']
interval = data.loc[:, 'time_interval_d']
date_daily = daily_data['date'].to_list()
precip_mm = daily_data['Sum(Precip_mm)'].tolist()
temperature = daily_data['Mean(airtemp_C)'].tolist()

dates = convert_datetime(date_daily)
date_years = [x.year for x in dates]
years_unique = list(set(date_years))
summer_precip = [0] * len(years_unique)
winter_precip = [0] * len(years_unique)
for i in range(len(dates)):
    year = dates[i].year
    month = int(dates[i].strftime('%m'))
    if month < 10:
        if year - 1 not in years_unique:
            continue
        index = years_unique.index(year - 1)
        if month < 5:
            winter_precip[index] += precip_mm[i]
        else:
            summer_precip[index] += precip_mm[i]
    else:
        index = years_unique.index(year)
        winter_precip[index] += precip_mm[i]

precip_df = pd.DataFrame({'year': years_unique, 'summer': summer_precip, 'winter': winter_precip})

# EMS with All Rietholzbach Catchment discharge data
stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
q_all = daily_data['Summed_RHB_discharge_mm'].tolist()
fluxes_all, iso_data_all, pwt_all, qwt_all, years_all = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope, date_daily, precip_mm, q_all, temperature)

# EMS with Upper Rietholzbach discharge data
stream_isotope_upp = data.loc[:, 'Upper_RHB_18O_combined']
fluxes_upp, iso_data_upp, pwt_upp, qwt_upp, years_upp = preprocessing(sampling_dates, precip_isotope, interval,
                                            stream_isotope_upp, date_daily, precip_mm, q_all, temperature)

# EMS with lysimeter seepage data
isotope_lysimeter = data.loc[:, 'Lysimeter_18O_combined']
lysimeter_seepage = daily_data['lys_seep_mm/day'].tolist()
fluxes_lys, iso_data_lys, pwt_lys, qwt_lys, years_lys = preprocessing(sampling_dates, precip_isotope, interval,
                                            isotope_lysimeter, date_daily, precip_mm, lysimeter_seepage, temperature)


# Figure 1 Isotope Values
plot_del_figure(q_all, stream_isotope, sampling_dates, date_daily, stream_isotope_upp, lysimeter_seepage,
                    isotope_lysimeter, precip_mm, precip_isotope, interval, iso_data_all, qwt_all, iso_data_upp,
                    qwt_upp, iso_data_lys, qwt_lys)


# Figure 2 Split Diagrams
longterm_all, lt_table_all = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, range(len(iso_data_all)))

longterm_upp, lt_table_upp = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, range(len(iso_data_upp)))

longterm_lys, lt_table_lys = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, range(len(iso_data_lys)))

fig, axs = plt.subplots(1, 3, figsize=(25, 10))
diagram_in_out(axs[0], lt_table_all, longterm_all, 'All RHB', Pw_ET_amt=80, Pw_ET_pct=80, Ps_Qs_pct=-35)
diagram_in_out(axs[1], lt_table_upp, longterm_upp, 'Upper RHB', Pw_ET_amt=20, Pw_ET_pct=20, Ps_Qs_pct=-40)
diagram_in_out(axs[2], lt_table_lys, longterm_lys, 'Lysimeter', Pw_ET_amt=20, Pw_ET_pct=20, Ps_Qs_pct=-30)
fig.tight_layout()
plt.show()


# Figure 3 Split Diagrams Dividing Years by Median Summer Precipitation
fig, axs = plt.subplots(3, 2, figsize=(18, 26))
year_indices = calc_year_indices(precip_df, years_all, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 0], table, df, 'All RHB Ps >= median', Pw_ET_amt=85, Pw_ET_pct=85, Ps_Qs_amt=-40, Ps_Qs_pct=-45)

year_indices = calc_year_indices(precip_df, years_all, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 1], table, df, 'All RHB Ps <= median', Pw_ET_amt=40, Pw_ET_pct=80, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-45, Ps_Qw_pct=-45)

year_indices = calc_year_indices(precip_df, years_upp, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Upper RHB Ps >= median', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30,
               Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Upper RHB Ps <= median', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '>= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 0], table, df, 'Lysimeter Ps >= median', Pw_ET_amt=100, Pw_ET_pct=100, Ps_Qs_amt=15,
               Ps_Qs_pct=-30)

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '<= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 1], table, df, 'Lysimeter Ps <= median', Pw_ET_amt=-10, Pw_ET_pct=-15, Ps_Qs_pct=-30,
               Pw_Qs_pct=-25, Ps_Qw_pct=20)

fig.tight_layout()
plt.show()


# Figure 4 Split Diagrams Dividing Years by Median Winter Precipitation
fig, axs = plt.subplots(3, 2, figsize=(18, 26))

year_indices = calc_year_indices(precip_df, years_all, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 0], table, df, 'All RHB Pw >= median', Pw_ET_amt=10, Pw_ET_pct=40, Ps_Qs_pct=-40, Pw_Qs_amt=-20,
               Pw_Qs_pct=-30)

year_indices = calc_year_indices(precip_df, years_all, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 1], table, df, 'All RHB Pw <= median', space=120, Ps_ET_pct=30, Pw_ET_amt=220, Pw_ET_pct=250,
               Ps_Qs_pct=-10, Ps_Qw_pct=-50)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Upper RHB Pw >= median', Pw_ET_pct=20, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Upper RHB Pw <= median', Ps_ET_pct=50, Pw_ET_amt=85, Pw_ET_pct=120,
               Ps_Qs_amt=-30, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '>= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 0], table, df, 'Lysimeter Pw >= median', Ps_ET_pct=20, Pw_ET_amt=-20, Pw_ET_pct=-10,
               Ps_Qs_amt=20, Ps_Qs_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '<= median')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 1], table, df, 'Lysimeter Pw <= median', space=120, Ps_ET_pct=50, Pw_ET_amt=185,
               Pw_ET_pct=170, Ps_Qs_pct=-50)

fig.tight_layout()
plt.show()


# Figure 5 Bootstrapping Results
X_axis = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660,
          670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880,
          890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]

plot_panels_q(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, X_axis, 'Upper RHB')

# Figure 6
evapotranspiration = pd.read_csv(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\Evapotranspiration_Hirschi.csv')

df_upp_et = plot_panels_et(years_upp, iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, evapotranspiration, X_axis,
                           'Upper RHB')
plot_correlations(df_upp_et, 'P_s', 'Upper RHB')
plot_correlations(df_upp_et, 'P_w', 'Upper RHB')


# Figure showing split diagram for years with dry/wet summers following dry/wet summers
fig, axs = plt.subplots(3, 2, figsize=(18, 26))

year_indices = calc_year_indices(precip_df, years_all, 'summer', '>= median x2')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 0], table, df, 'All RHB Ps >= median x2', Pw_ET_amt=85, Pw_ET_pct=85, Ps_Qs_amt=-40,
               Ps_Qs_pct=-45)

year_indices = calc_year_indices(precip_df, years_all, 'summer', '<= median x2')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 1], table, df, 'All RHB Ps <= median', Pw_ET_amt=40, Pw_ET_pct=80, Ps_Qs_pct=-30, Pw_Qs_amt=-20,
               Pw_Qs_pct=-45, Ps_Qw_pct=-45)

year_indices = calc_year_indices(precip_df, years_upp, 'summer', '>= median x2')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Upper RHB Ps >= median x2', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30, Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'summer', '<= median x2')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Upper RHB Ps <= median x2', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20, Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '>= median x2')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 0], table, df, 'Lysimeter Ps >= median x2', Pw_ET_amt=100, Pw_ET_pct=100, Ps_Qs_amt=15, Ps_Qs_pct=-30)

year_indices = calc_year_indices(precip_df, years_lys, 'summer', '<= median x2')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 1], table, df, 'Lysimeter Ps <= median x2', Pw_ET_amt=-10, Pw_ET_pct=-15, Ps_Qs_pct=-30, Pw_Qs_pct=-25, Ps_Qw_pct=20)

fig.tight_layout()
plt.show()


# Figure showing split diagram for years with dry/wet winters following dry/wet winters
fig, axs = plt.subplots(3, 2, figsize=(18, 26))

year_indices = calc_year_indices(precip_df, years_all, 'winter', '>= median x2')
df, table = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 0], table, df, 'All RHB Pw >= median x2', Pw_ET_amt=85, Pw_ET_pct=85, Ps_Qs_amt=-40, Ps_Qs_pct=-45)

year_indices = calc_year_indices(precip_df, years_all, 'winter', '<= median x2')
df, table  = multi_year_endsplit(iso_data_all, fluxes_all, pwt_all, qwt_all, year_indices)
diagram_in_out(axs[0, 1], table, df, 'All RHB Pw <= median', Pw_ET_amt=40, Pw_ET_pct=80, Ps_Qs_pct=-30, Pw_Qs_amt=-20, Pw_Qs_pct=-45, Ps_Qw_pct=-45)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '>= median x2')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 0], table, df, 'Upper RHB Pw >= median x2', Pw_ET_amt=60, Pw_ET_pct=70, Ps_Qs_amt=-30, Ps_Qs_pct=-40, Pw_Qs_amt=-20)

year_indices = calc_year_indices(precip_df, years_upp, 'winter', '<= median x2')
df, table = multi_year_endsplit(iso_data_upp, fluxes_upp, pwt_upp, qwt_upp, year_indices)
diagram_in_out(axs[1, 1], table, df, 'Upper RHB Pw <= median x2', Pw_ET_pct=35, Ps_Qs_pct=-30, Pw_Qs_amt=-20, Pw_Qs_pct=-35, Ps_Qw_pct=-20)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '>= median x2')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 0], table, df, 'Lysimeter Pw >= median x2', Pw_ET_amt=100, Pw_ET_pct=100, Ps_Qs_amt=15, Ps_Qs_pct=-30)

year_indices = calc_year_indices(precip_df, years_lys, 'winter', '<= median x2')
df, table = multi_year_endsplit(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, year_indices)
diagram_in_out(axs[2, 1], table, df, 'Lysimeter Pw <= median x2', Pw_ET_amt=-10, Pw_ET_pct=-15, Ps_Qs_pct=-30, Pw_Qs_pct=-25, Ps_Qw_pct=20)

fig.tight_layout()
plt.show()


# Plot bootstrapping results and variable correlations for All RHB
plot_panels_q(iso_data_all, fluxes_all, pwt_all, qwt_all, X_axis, 'All RHB')
df_all_et = plot_panels_et(years_all, iso_data_all, fluxes_all, pwt_all, qwt_all, evapotranspiration, X_axis, 'All RHB')
plot_correlations(df_all_et, 'P_s', 'All RHB')
plot_correlations(df_all_et, 'P_w', 'All RHB')


# Plot bootstrapping results and variable correlations for Lysimeter
plot_panels_q(iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, X_axis, 'Lysimeter')
df_lys_et = plot_panels_et(years_lys, iso_data_lys, fluxes_lys, pwt_lys, qwt_lys, evapotranspiration, X_axis, 'Lysimeter')
plot_correlations(df_lys_et, 'P_s', 'Lysimeter')
plot_correlations(df_lys_et, 'P_w', 'Lysimeter')


