import math
from datetime import datetime as dt

# Convert date strings to datetime objects
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

# List all years represented by dates
def list_years(dates):
    aux_list = []
    years = []
    for year in dates:
        if int(year.strftime('%Y')) not in aux_list:
            years.append(int(year.strftime('%Y')))
            aux_list.append(int(year.strftime('%Y')))
    return years

# Find the first (or last) month and year sampled
def find_start_sampling(dates, dates_daily, Pdel, P, Qdel, Q, side=''):
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
            return month_flux, year_flux
        else:
            return month_iso, year_iso
    elif side == 'end':
        if year_flux < year_iso or year_flux == year_iso and month_flux < month_iso:
            return month_flux, year_flux
        else:
            return month_iso, year_iso
    else:
        return print("Choose start or end of sampling period")

# List dates outside of a fully sampled hydrologic year which runs from October to September
def list_unusable_dates(dates, first_month, first_year_index, last_month, last_year_index, years):
    no_count_date = []
    for i in range(len(dates)):
        year_index = years.index(int(dates[i].strftime('%Y')))
        month = int(dates[i].strftime('%m'))
        if first_month > 10:
            if year_index <= first_year_index or month < 10 and year_index == first_year_index + 1:
                no_count_date.append(dates[i])
        elif first_month <= 10:
            if month < 10 and year_index == first_year_index or year_index < first_year_index:
                no_count_date.append(dates[i])
        if last_month >= 10:
            if month >= 10 and year_index == last_year_index or year_index > last_year_index:
                no_count_date.append(dates[i])
        elif last_month < 10:
            if month >= 10 and year_index == last_year_index - 1 or year_index >= last_year_index:
                no_count_date.append(dates[i])
    return no_count_date

# Remove the dates listed in ListUnusableDates and split the flux data into each hydrologic year
def split_fluxes_by_hydro_year(dates, years, no_count_date, precipitation, runoff, temperature):
    flux_data = [[] for _ in years]
    for y in range(len(years)):
         flux_data[y] = {'year': years[y], 'dates': [], 'P': [], 'Pcat': [], 'Q': [], 'Qcat': [], 'Tcat': []}
    for i in range(len(dates)):
        if dates[i] in no_count_date:
            continue
        month = int(dates[i].strftime('%m'))
        index = years.index(int(dates[i].strftime('%Y')))
        if month < 10:
            flux_data[index - 1]['dates'].append(dates[i])
            flux_data[index - 1]['P'].append(precipitation[i])
            flux_data[index - 1]['Q'].append(runoff[i])
            if month < 5:
                flux_data[index - 1]['Pcat'].append("winter")
                flux_data[index - 1]['Qcat'].append("winter")
            else:
                flux_data[index - 1]['Pcat'].append("summer")
                flux_data[index - 1]['Qcat'].append("summer")
            if temperature[i] <= 0:
                flux_data[index - 1]['Tcat'].append('snow')
            else:
                flux_data[index - 1]['Tcat'].append('rain')
        else:
            flux_data[index]['dates'].append(dates[i])
            flux_data[index]['P'].append(precipitation[i])
            flux_data[index]['Q'].append(runoff[i])
            flux_data[index]['Pcat'].append("winter")
            flux_data[index]['Qcat'].append("winter")
            if temperature[i] <= 0:
                flux_data[index]['Tcat'].append('snow')
            else:
                flux_data[index]['Tcat'].append('rain')
    return flux_data

# Remove the dates listed above and split isotope data into each year
# If the sampling period spans two seasons, add a date for the last day of the season to split up the fluxes by season
def split_isotopes_by_hydro_year(dates, intervals, years, no_count_date, precipitation, runoff, first_year):
    date_by_year = [[] for _ in years]
    precip_d_year = [[] for _ in years]
    pdelcat_with_nan = [[] for _ in years]
    runoff_d_year = [[] for _ in years]
    qdelcat_with_nan = [[] for _ in years]
    interval_by_year = [[] for _ in years]

    for i in range(len(dates)):
        if dates[i] in no_count_date:
            continue
        month = int(dates[i].strftime('%m'))
        days_in_month = int(dates[i].strftime('%d'))
        year = dates[i].strftime('%Y')
        index = years.index(int(year))
        interval = int(intervals[i])
        if month < 10:
            if month == 5 and days_in_month < interval:
                date_by_year[index - 1].append(dt(int(year), 4, 30))
                date_by_year[index - 1].append(dates[i])
                precip_d_year[index - 1].append(precipitation[i])
                precip_d_year[index - 1].append(precipitation[i])
                pdelcat_with_nan[index - 1].append("winter")
                pdelcat_with_nan[index - 1].append("summer")
                runoff_d_year[index - 1].append(runoff[i])
                runoff_d_year[index - 1].append(runoff[i])
                qdelcat_with_nan[index - 1].append("winter")
                qdelcat_with_nan[index - 1].append("summer")
                interval_by_year[index - 1].append(interval - days_in_month)
                interval_by_year[index - 1].append(days_in_month)
            else:
                date_by_year[index - 1].append(dates[i])
                precip_d_year[index - 1].append(precipitation[i])
                runoff_d_year[index - 1].append(runoff[i])
                interval_by_year[index - 1].append(interval)
                if month < 5:
                    pdelcat_with_nan[index - 1].append("winter")
                    qdelcat_with_nan[index - 1].append("winter")
                else:
                    pdelcat_with_nan[index - 1].append("summer")
                    qdelcat_with_nan[index - 1].append("summer")
        else:
            if month == 10 and days_in_month < interval:
                date_by_year[index].append(dates[i])
                precip_d_year[index].append(precipitation[i])
                pdelcat_with_nan[index].append("winter")
                runoff_d_year[index].append(runoff[i])
                qdelcat_with_nan[index].append("winter")
                interval_by_year[index].append(days_in_month)
                if year == first_year:
                    continue
                date_by_year[index-1].append(dt(int(year), 9, 30))
                precip_d_year[index - 1].append(precipitation[i])
                pdelcat_with_nan[index - 1].append("summer")
                runoff_d_year[index - 1].append(runoff[i])
                qdelcat_with_nan[index - 1].append("summer")
                interval_by_year[index - 1].append(interval - days_in_month)
            else:
                date_by_year[index].append(dates[i])
                precip_d_year[index].append(precipitation[i])
                pdelcat_with_nan[index].append("winter")
                runoff_d_year[index].append(runoff[i])
                qdelcat_with_nan[index].append("winter")
                interval_by_year[index].append(interval)
    return date_by_year, precip_d_year, pdelcat_with_nan, runoff_d_year, qdelcat_with_nan, interval_by_year