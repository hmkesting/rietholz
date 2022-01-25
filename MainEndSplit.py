import pandas as pd
from preprocessing import ConvertDateTime, ListYears, FindStartOfSampling, ListUnusableDates, IdentifyMissingData, SplitFluxesByHydrologicYear, SplitIsotopesByHydrologicYear
from cleaning import SumPrecipitationAndRunoff, RemoveNanSamples
from calculations import LysimeterUndercatch, EndSplit
from plot import PlotFraction, PlotAmount

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

def WorkFlowEndSplit(dates, Precip_isotope, Interval, Stream_isotope, dates_daily, P, Q, Watershed, lag_years, temperature, instrument='none'):
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

    date_daily_by_year, P_by_year, Pcat, Q_by_year, Qcat, Precip_cat = SplitFluxesByHydrologicYear(date_daily, years_daily,
                                                                                       no_count_date_daily, P, Q, temperature)

    date_by_year, precip_d_year, Pdelcat_with_nan, runoff_d_year, Qdelcat_with_nan, interval_by_year = SplitIsotopesByHydrologicYear(Sampling_dates, Interval, years, no_count_date, Precip_isotope, Stream_isotope)

    #print('Missing isotope data')
    # The following line shows missing isotope data if relevant lines in IdentifyMissingData are uncommented
    #remove_years = IdentifyMissingData(date_by_year, precip_d_year, runoff_d_year, interval_by_year, years)

    interval = [[] for _ in date_daily_by_year]
    for i in range(len(date_daily_by_year)):
        for d in range(len(date_daily_by_year[i])):
            interval[i].append(1)

    remove_years_daily = IdentifyMissingData(date_daily_by_year, P_by_year, Q_by_year, interval, years_daily)
    if remove_years_daily:
        print(' ')
        print('Remove years', remove_years_daily, 'due to missing flux data')

    Pwt, Qwt = SumPrecipitationAndRunoff(years, date_by_year, date_daily_by_year, P_by_year, Q_by_year, precip_d_year, runoff_d_year)

    Qdel, Qdelcat = RemoveNanSamples(years, runoff_d_year, Qdelcat_with_nan)

    Pdel, Pdelcat = RemoveNanSamples(years, precip_d_year, Pdelcat_with_nan)

    # Analysis using values from year of interest (assumes no lag or mixing)

    f_ET_from_summer = []
    f_ET_se = []
    ET = []
    ET_se = []
    summer_P = []
    summer_P_se = []

    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            f_ET_from_summer, f_ET_se, ET, ET_se, summer_P, summer_P_se = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], P_by_year[i], Q_by_year[i], Pcat[i], Qcat[i], f_ET_from_summer, f_ET_se, ET, ET_se, summer_P, summer_P_se)

    PlotFraction(summer_P, summer_P_se, f_ET_from_summer, f_ET_se, "Current", Watershed)

    # PlotAmount function plots summer precipitation by the total amount of evapotranspiration
    PlotAmount(summer_P, summer_P_se, ET, ET_se, "Current", Watershed)

    # Lagged flow analysis
    # Repeat end-member splitting, but use previous year's precipitation isotope values, keeping fluxes from year of interest

    f_ET_from_summer_lag = []
    f_ET_se_lag = []
    ET_lag = []
    ET_se_lag = []
    summer_P_lag = []
    summer_P_se_lag = []

    Qdel_lag = Qdel[lag_years:]
    Qwt_lag = Qwt[lag_years:]
    Qdelcat_lag = Qdelcat[lag_years:]
    P_lag = P_by_year[lag_years:]
    Q_lag = Q_by_year[lag_years:]
    Pcat_lag = Pcat[lag_years:]
    Qcat_lag = Qcat[lag_years:]
    years_lag = years[lag_years:]

    for i in range(len(Qwt_lag)):
        if Pwt[i] and Qwt_lag[i] and years_lag[i] not in remove_years_daily:
            f_ET_from_summer_lag, f_ET_se_lag, ET_lag, ET_se_lag, summer_P_lag, summer_P_se_lag = EndSplit(Pdel[i], Qdel_lag[i], Pwt[i], Qwt_lag[i], Pdelcat[i], Qdelcat_lag[i], P_lag[i], Q_lag[i], Pcat_lag[i], Qcat_lag[i], f_ET_from_summer_lag, f_ET_se_lag, ET_lag, ET_se_lag, summer_P_lag, summer_P_se_lag)

    PlotFraction(summer_P_lag, summer_P_se_lag, f_ET_from_summer_lag, f_ET_se_lag, "Lagged", Watershed)

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

    f_ET_from_summer_mixed = []
    f_ET_se_mixed = []
    ET_mixed = []
    ET_se_mixed = []
    summer_P_mixed = []
    summer_P_se_mixed = []
    for i in range(len(years)):
        if Pwt[i] and years[i] not in remove_years_daily:
            f_ET_from_summer_mixed, f_ET_se_mixed, ET_mixed, ET_se_mixed, summer_P_mixed, summer_P_se_mixed = EndSplit(isotope_vals, Qdel[i], weights, Qwt[i], season, Qdelcat[i], P_by_year[i], Q_by_year[i], Pcat[i], Qcat[i], f_ET_from_summer_mixed, f_ET_se_mixed, ET_mixed, ET_se_mixed, summer_P_mixed, summer_P_se_mixed)

    PlotFraction(summer_P_mixed, summer_P_se_mixed, f_ET_from_summer_mixed, f_ET_se_mixed, "Average", Watershed)

    if instrument == 'lysimeter':

        print('Lysimeter Undercatch Sensitivity Analysis')

        print('f_ET_from_summer: ', f_ET_from_summer)

        # Adjust both rain and snowfall for undercatch

        precipitation_adjusted = LysimeterUndercatch(P_by_year, Precip_cat, category='both')

        Pwt, Qwt = SumPrecipitationAndRunoff(years, date_by_year, date_daily_by_year, precipitation_adjusted, Q_by_year,
                                             precip_d_year, runoff_d_year)

        f_ET_from_summer_adjusted = []
        f_ET_from_summer_adjusted_se = []
        ET_adjusted = []
        ET_adjusted_se = []
        summer_P_adjusted = []
        summer_P_adjusted_se = []

        for i in range(len(years)):
            if Pwt[i] and years[i] not in remove_years_daily:
                f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se, summer_P_adjusted, summer_P_adjusted_se = EndSplit(Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], precipitation_adjusted[i], Q_by_year[i], Pcat[i], Qcat[i], f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se, summer_P_adjusted, summer_P_adjusted_se)

        PlotFraction(summer_P_adjusted, summer_P_adjusted_se, f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, "Current (rain and snow adjusted)", Watershed)
        PlotAmount(summer_P_adjusted, summer_P_adjusted_se, ET_adjusted, ET_adjusted_se, "Current (rain and snow adjusted)", Watershed)

        print('Rain and snow adjusted f_ET_from_summer: ', f_ET_from_summer_adjusted)

        # Adjust only snowfall for undercatch

        precipitation_adjusted = LysimeterUndercatch(P_by_year, Precip_cat, category='rain')

        Pwt, Qwt = SumPrecipitationAndRunoff(years, date_by_year, date_daily_by_year, precipitation_adjusted, Q_by_year,
                                             precip_d_year, runoff_d_year)

        f_ET_from_summer_adjusted = []
        f_ET_from_summer_adjusted_se = []
        ET_adjusted = []
        ET_adjusted_se = []
        summer_P_adjusted = []
        summer_P_adjusted_se = []

        for i in range(len(years)):
            if Pwt[i] and years[i] not in remove_years_daily:
                f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se, summer_P_adjusted, summer_P_adjusted_se = EndSplit(
                    Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], precipitation_adjusted[i], Q_by_year[i],
                    Pcat[i], Qcat[i], f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se,
                    summer_P_adjusted, summer_P_adjusted_se)

        PlotFraction(summer_P_adjusted, summer_P_adjusted_se, f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se,
                     "Current (rain adjusted)", Watershed)
        PlotAmount(summer_P_adjusted, summer_P_adjusted_se, ET_adjusted, ET_adjusted_se,
                   "Current (rain adjusted)", Watershed)

        print('Rain adjusted f_ET_from_summer: ', f_ET_from_summer_adjusted)

        # Adjust only snowfall for undercatch

        precipitation_adjusted = LysimeterUndercatch(P_by_year, Precip_cat, category='snow')

        Pwt, Qwt = SumPrecipitationAndRunoff(years, date_by_year, date_daily_by_year, precipitation_adjusted, Q_by_year,
                                             precip_d_year, runoff_d_year)

        f_ET_from_summer_adjusted = []
        f_ET_from_summer_adjusted_se = []
        ET_adjusted = []
        ET_adjusted_se = []
        summer_P_adjusted = []
        summer_P_adjusted_se = []
        for i in range(len(years)):
            if Pwt[i] and years[i] not in remove_years_daily:
                f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se, summer_P_adjusted, summer_P_adjusted_se = EndSplit(
                    Pdel[i], Qdel[i], Pwt[i], Qwt[i], Pdelcat[i], Qdelcat[i], precipitation_adjusted[i], Q_by_year[i],
                    Pcat[i], Qcat[i], f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se, ET_adjusted, ET_adjusted_se,
                    summer_P_adjusted, summer_P_adjusted_se)

        PlotFraction(summer_P_adjusted, summer_P_adjusted_se, f_ET_from_summer_adjusted, f_ET_from_summer_adjusted_se,
                     "Current (snow adjusted)", Watershed)
        PlotAmount(summer_P_adjusted, summer_P_adjusted_se, ET_adjusted, ET_adjusted_se,
                   "Current (snow adjusted)", Watershed)

        print('Snow adjusted f_ET_from_summer: ', f_ET_from_summer_adjusted)

# All Rietholzbach Catchment

Stream_isotope = data.loc[:, 'Main_gauge_18O_combined']
Q = daily_data['Summed_RHB_discharge_mm'].tolist()

WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Stream_isotope, date_daily, P, Q, "All", 1, temperature)

# Upper Rietholzbach

Stream_isotope_upper = data.loc[:, 'Upper_RHB_18O_combined']
Q_upper = daily_data['SummedDischarge(upper_RHB_mm/h)'].tolist()

WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Stream_isotope_upper, date_daily, P, Q_upper, "Upper", 2, temperature)

# Lysimeter

Isotope_lysimeter_seepage = data.loc[:, 'Lysimeter_18O_combined']
Amount_lysimeter_seepage = daily_data['lys_seep_mm/day']

WorkFlowEndSplit(Sampling_dates, Precip_isotope, Interval, Isotope_lysimeter_seepage, date_daily, P, Amount_lysimeter_seepage, "Lysimeter", 1, temperature, instrument='lysimeter')



