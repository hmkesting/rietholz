from datetime import datetime as dt
import math
import copy
import numpy as np


def convert_datetime(dates):
    converted_dates = []
    for i in dates:
        if i == '':
            raise Exception('Empty string in dates')
        elif int(i.split('/')[0]) > 12:
            raise Exception('Date must fit format month/day/year')
        elif len(i.split('/')[2]) == 4:
            converted = dt.strptime(i, '%m/%d/%Y')
            converted_dates.append(converted)
        elif len(i.split('/')[2]) == 2:
            converted = dt.strptime(i, '%m/%d/%y')
            converted_dates.append(converted)
        else:
            raise Exception('Date must fit format mm/dd/yy or mm/dd/yyyy')
    return converted_dates


def list_years(dates):
    aux_list = []
    years = []
    for year in dates:
        if int(year.strftime('%Y')) not in aux_list:
            years.append(int(year.strftime('%Y')))
            aux_list.append(int(year.strftime('%Y')))
    return years


# Determine which years are fully sampled with both winter and summer having precipitation and runoff data
def find_range_sampling(dates, dates_daily, Pdel, P, Qdel, Q, cutoff_month, side=''):
    for i in range(len(dates)):
        if math.isnan(Pdel[i]) or math.isnan(Qdel[i]):
            continue
        else:
            month_iso = int(dates[i].strftime('%m'))
            year_iso = int(dates[i].strftime('%Y'))
            break
    for i in range(len(dates_daily)):
        if math.isnan(P[i]) or math.isnan(Q[i]):
            continue
        else:
            month_flux = int(dates_daily[i].strftime('%m'))
            year_flux = int(dates_daily[i].strftime('%Y'))
            break
    if side == 'start':
        if year_flux > year_iso or year_flux == year_iso and month_flux > month_iso:
            if month_flux <= cutoff_month:
                return year_flux
            if month_flux > cutoff_month:
                return year_flux + 1
        else:
            if month_iso <= cutoff_month:
                return year_iso
            if month_iso > cutoff_month:
                return year_iso + 1
    elif side == 'end':
        if year_flux < year_iso or year_flux == year_iso and month_flux < month_iso:
            if month_flux >= cutoff_month:
                return year_flux
            if month_flux < cutoff_month:
                return year_flux - 1
        else:
            if month_iso >= cutoff_month:
                return year_iso
            if month_iso < cutoff_month:
                return year_iso - 1
    else:
        raise Exception("Choose start or end of sampling period")


def list_unusable_dates(dates, first_month, first_year, last_month, last_year):
    no_count_date = []
    for i in range(len(dates)):
        year = int(dates[i].strftime('%Y'))
        month = int(dates[i].strftime('%m'))
        if year < first_year or year == first_year and month < first_month:
            no_count_date.append(dates[i])
        if year > last_year or year == last_year and month > last_month:
            no_count_date.append(dates[i])
    return no_count_date


def split_fluxes_by_hydro_year(dates, years, start_winter, start_summer, no_count_date, precipitation, runoff, temperature, undercatch_type=None):
    flux_data = [[] for _ in years]
    for y in range(len(years)):
         flux_data[y] = {'year': years[y], 'dates': [], 'P': [], 'Pcat': [], 'Q': [], 'Qcat': [], 'Tcat': [], 'T': []}
    for i in range(len(dates)):
        if dates[i] in no_count_date:
            continue
        month = int(dates[i].strftime('%m'))
        if month < start_winter:
            index = years.index(int(dates[i].strftime('%Y')) - 1)
            if month < start_summer:
                flux_data[index]['Pcat'].append("winter")
                flux_data[index]['Qcat'].append("winter")
            else:
                flux_data[index]['Pcat'].append("summer")
                flux_data[index]['Qcat'].append("summer")
            if temperature[i] <= 2:
                flux_data[index]['Tcat'].append('snow')
            elif np.isnan(temperature[i]):
                flux_data[index]['Tcat'].append('unknown')
            else:
                flux_data[index]['Tcat'].append('rain')
        else:
            index = years.index(int(dates[i].strftime('%Y')))
            flux_data[index]['Pcat'].append("winter")
            flux_data[index]['Qcat'].append("winter")
            if temperature[i] <= 0:
                flux_data[index]['Tcat'].append('snow')
            elif np.isnan(temperature[i]):
                flux_data[index]['Tcat'].append('unknown')
            else:
                flux_data[index]['Tcat'].append('rain')
        flux_data[index]['dates'].append(dates[i])
        flux_data[index]['P'].append(precipitation[i])
        flux_data[index]['Q'].append(runoff[i])
        flux_data[index]['T'].append(temperature[i])
    if undercatch_type == None:
        return flux_data
    elif undercatch_type == 'rain':
        for index in range(len(flux_data)):
            for i in range(len(flux_data[index]['Tcat'])):
                if flux_data[index]['Tcat'][i] == 'rain':
                    flux_data[index]['P'][i] = flux_data[index]['P'][i] * 1.15
        return flux_data
    elif undercatch_type == 'snow':
        for index in range(len(flux_data)):
            for i in range(len(flux_data[index]['Tcat'])):
                if flux_data[index]['Tcat'][i] == 'snow':
                    flux_data[index]['P'][i] = flux_data[index]['P'][i] * 1.5
        return flux_data
    elif undercatch_type == 'both':
        for index in range(len(flux_data)):
            for i in range(len(flux_data[index]['Tcat'])):
                if flux_data[index]['Tcat'][i] == 'rain':
                    flux_data[index]['P'][i] = flux_data[index]['P'][i] * 1.15
                if flux_data[index]['Tcat'][i] == 'snow':
                    flux_data[index]['P'][i] = flux_data[index]['P'][i] * 1.5
        return flux_data


def split_isotopes_by_hydro_year(dates, intervals, years, no_count_date, precipitation, runoff):
    iso_data = [[] for _ in years]
    for y in range(len(years)):
        iso_data[y] = {'year': years[y], 'Pdel_dates': [], 'Qdel_dates': [], 'Pdel': [], 'Pdelcat': [], 'Qdel': [],
                       'Qdelcat': []}
    for i in range(len(dates)):
        if dates[i] in no_count_date:
            continue
        month = int(dates[i].strftime('%m'))
        days_in_month = int(dates[i].strftime('%d'))
        year = int(dates[i].strftime('%Y'))
        interval = int(intervals[i])
        if month < 10:
            index = years.index(int(year) - 1)
        else:
            index = years.index(int(year))
        iso_data[index]['Pdel'].append(precipitation[i])
        iso_data[index]['Qdel'].append(runoff[i])
        if month == 5 and days_in_month < interval:
            iso_data[index]['Pdel_dates'].append(dt(int(year), 4, 30))
            iso_data[index]['Qdel_dates'].append(dt(int(year), 4, 30))
            iso_data[index]['Pdel'].append(precipitation[i])
            iso_data[index]['Pdelcat'].append("winter")
            iso_data[index]['Pdelcat'].append("summer")
            iso_data[index]['Qdel'].append(runoff[i])
            iso_data[index]['Qdelcat'].append("winter")
            iso_data[index]['Qdelcat'].append("summer")
        elif month == 10 and days_in_month < interval:
            if year != years[0]:
                iso_data[index - 1]['Pdel_dates'].append(dt(int(year), 9, 30))
                iso_data[index - 1]['Qdel_dates'].append(dt(int(year), 9, 30))
                iso_data[index - 1]['Pdelcat'].append("summer")
                iso_data[index - 1]['Qdelcat'].append("summer")
                iso_data[index - 1]['Pdel'].append(precipitation[i])
                iso_data[index - 1]['Qdel'].append(runoff[i])
            iso_data[index]['Pdelcat'].append("winter")
            iso_data[index]['Qdelcat'].append("winter")
        elif month < 5 or month >= 10:
            iso_data[index]['Pdelcat'].append("winter")
            iso_data[index]['Qdelcat'].append("winter")
        else:
            iso_data[index]['Pdelcat'].append("summer")
            iso_data[index]['Qdelcat'].append("summer")
        iso_data[index]['Pdel_dates'].append(dates[i])
        iso_data[index]['Qdel_dates'].append(dates[i])
    return iso_data


def sum_precipitation_and_runoff(iso_data, fluxes):
    pwt = [[] for _ in fluxes]
    qwt = [[] for _ in fluxes]
    sum_precip = 0
    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            sum_precip += fluxes[i]['P'][d]
            daily_stream = fluxes[i]['Q'][d]
            if fluxes[i]['dates'][d] in iso_data[i]['Pdel_dates']:
                index = iso_data[i]['Pdel_dates'].index(fluxes[i]['dates'][d])
                if not math.isnan(iso_data[i]['Pdel'][index]):
                    pwt[i].append(sum_precip)
                sum_precip = 0
            if fluxes[i]['dates'][d] in iso_data[i]['Qdel_dates']:
                index = iso_data[i]['Qdel_dates'].index(fluxes[i]['dates'][d])
                if not math.isnan(iso_data[i]['Qdel'][index]):
                    qwt[i].append(daily_stream)
                sum_precip = 0
    return pwt, qwt


def remove_nan_samples(iso_data_with_nans):
    iso_data = copy.deepcopy(iso_data_with_nans)
    for i in range(len(iso_data)):
        if len(iso_data[i]['Pdel']) != len(iso_data[i]['Qdel']):
            raise Exception('Pdel and Qdel must be the same length')
        Pdel_nan = []
        Qdel_nan = []
        for d in range(len(iso_data[i]['Pdel'])):
            if math.isnan(iso_data[i]['Pdel'][d]):
                Pdel_nan.append(d)
            if math.isnan(iso_data[i]['Qdel'][d]):
                Qdel_nan.append(d)
        for x in Pdel_nan[::-1]:
            iso_data[i]['Pdel'].pop(x)
            iso_data[i]['Pdelcat'].pop(x)
            iso_data[i]['Pdel_dates'].pop(x)
        for x in Qdel_nan[::-1]:
            iso_data[i]['Qdel'].pop(x)
            iso_data[i]['Qdelcat'].pop(x)
            iso_data[i]['Qdel_dates'].pop(x)
    return iso_data


def preprocessing(dates, precip_isotope, sampling_interval, stream_isotope, dates_daily, p, q, temperature, undercatch_type=None):
    # List dates by hydrologic year (October to September)
    sampling_dates = convert_datetime(dates)
    date_daily = convert_datetime(dates_daily)
    years = list_years(sampling_dates)
    start_month = 10
    end_month = 9
    start_summer = 5
    first_year = find_range_sampling(sampling_dates, date_daily, precip_isotope, p, stream_isotope, q, start_month,
                                       side='start')
    last_year = find_range_sampling(sampling_dates[::-1], date_daily[::-1], precip_isotope.values[::-1],
                                       p[::-1], stream_isotope.values[::-1], q[::-1], end_month, side='end')
    years = [x for x in years if x >= first_year and x < last_year]
    no_count_date = list_unusable_dates(sampling_dates, start_month, first_year, end_month, last_year)
    no_count_date_daily = list_unusable_dates(date_daily, start_month, first_year, end_month, last_year)

    # Organize all data by hydrologic year
    fluxes = split_fluxes_by_hydro_year(date_daily, years, start_month, start_summer, no_count_date_daily, p, q, temperature, undercatch_type)
    iso_data_with_nans = split_isotopes_by_hydro_year(sampling_dates, sampling_interval, years,
                                     no_count_date, precip_isotope, stream_isotope)
    pwt, qwt = sum_precipitation_and_runoff(iso_data_with_nans, fluxes)
    iso_data = remove_nan_samples(iso_data_with_nans)

    return fluxes, iso_data, pwt, qwt, years