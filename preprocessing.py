import math
from datetime import datetime as dt

# Convert date strings to datetime objects
def ConvertDateTime(dates):
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
def ListYears(dates):
    aux_list = []
    years = []
    for year in dates:
        if int(year.strftime('%Y')) not in aux_list:
            years.append(int(year.strftime('%Y')))
            aux_list.append(int(year.strftime('%Y')))
    return years

# Find the first (or last) month and year sampled
def FindStartOfSampling(dates, dates_daily, Pdel, P, Qdel, Q, side='start'):
    for i in range(len(dates)):
        if math.isnan(Pdel[i]) or math.isnan(Qdel[i]):
            continue
        else:
            month = int(dates[i].strftime('%m'))
            year = dates[i].strftime('%Y')
            break
    for i in range(len(dates_daily)):
        if math.isnan(P[i]) or math.isnan(Q[i]):
            continue
        else:
            month_flux = int(dates_daily[i].strftime('%m'))
            year_flux = dates_daily[i].strftime('%Y')
            break
    if side == 'start':
        if year_flux > year or year_flux == year and month_flux > month:
            return month_flux, year_flux
        else:
            return month, year
    elif side == 'end':
        if year_flux < year or year_flux == year and month_flux < month:
            return month_flux, year_flux
        else:
            return month, year
    else:
        return print("Choose start or end of sampling period")

# List dates outside of a fully sampled hydrologic year which runs from October to September
def ListUnusableDates(dates, first_month, first_year_index, last_month, last_year_index, years):
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

# Identify which years need to be removed due to a lack of runoff data (will not work for missing precipitation data)
def IdentifyMissingData(dates, precipitation, runoff, interval, years):
    aux_list_precip = [[] for _ in dates]
    count_list_precip = [[] for _ in dates]
    no_data_dates_precip = [[] for _ in dates]
    aux_list_runoff = [[] for _ in dates]
    count_list_runoff = [[] for _ in dates]
    no_data_dates_runoff = [[] for _ in dates]
    remove_years = []
    for i in range(len(dates)):
        for d in range(len(dates[i])):
            month_year = dates[i][d].strftime("%b '%y")
            if math.isnan(precipitation[i][d]) or math.isnan(runoff[i][d]):
                if math.isnan(precipitation[i][d]) and not math.isnan(runoff[i][d]):
                    if month_year not in aux_list_precip[i]:
                        aux_list_precip[i].append(month_year)
                        no_data_dates_precip[i].append(dates[i][d])
                        count_list_precip[i].append(interval[i][d])
                    else:
                        index = aux_list_precip[i].index(month_year)
                        no_data_dates_precip[i].append(dates[i][d])
                        count_list_precip[i][index] += interval[i][d]
                elif math.isnan(precipitation[i][d]) and math.isnan(runoff[i][d]):
                    if month_year not in aux_list_precip[i]:
                        aux_list_precip[i].append(month_year)
                        no_data_dates_precip[i].append(dates[i][d])
                        count_list_precip[i].append(interval[i][d])
                    else:
                        index = aux_list_precip[i].index(month_year)
                        no_data_dates_precip[i].append(dates[i][d])
                        count_list_precip[i][index] += interval[i][d]
                    if month_year not in aux_list_runoff[i]:
                        aux_list_runoff[i].append(month_year)
                        no_data_dates_runoff[i].append(dates[i][d])
                        count_list_runoff[i].append(interval[i][d])
                    else:
                        index = aux_list_runoff[i].index(month_year)
                        no_data_dates_runoff[i].append(dates[i][d])
                        count_list_runoff[i][index] += interval[i][d]
                elif not math.isnan(precipitation[i][d]) and math.isnan(runoff[i][d]):
                    if month_year not in aux_list_runoff[i]:
                        aux_list_runoff[i].append(month_year)
                        no_data_dates_runoff[i].append(dates[i][d])
                        count_list_runoff[i].append(interval[i][d])
                    else:
                        index = aux_list_runoff[i].index(month_year)
                        no_data_dates_runoff[i].append(dates[i][d])
                        count_list_runoff[i][index] += interval[i][d]
    # The commented lines below print the number of days per year missing precipitation and runoff data as well as which months are missing data
    #for i in range(len(aux_list_precip)):
        #if aux_list_precip[i]:
            #print(sum(count_list_precip[i]), 'days of precipitation values missing in', aux_list_precip[i])
    for i in range(len(aux_list_runoff)):
        if aux_list_runoff[i]:
            #print(sum(count_list_runoff[i]), 'days of runoff values missing in', aux_list_runoff[i])
            if sum(count_list_runoff[i]) > 5:
                remove_years.append(years[i])
    return remove_years

# Remove the dates listed in ListUnusableDates and split the flux data into each hydrologic year
def SplitFluxesByHydrologicYear(dates, years, no_count_date, precipitation, runoff, temperature):
    date_daily_by_year = [[] for _ in years]
    P_by_year = [[] for _ in years]
    Q_by_year = [[] for _ in years]
    Pcat = [[] for _ in years]
    Qcat = [[] for _ in years]
    Precip_cat = [[] for _ in years]
    for i in range(len(dates)):
        if dates[i] in no_count_date:
            continue
        month = int(dates[i].strftime('%m'))
        index = years.index(int(dates[i].strftime('%Y')))
        if month < 10:
            date_daily_by_year[index - 1].append(dates[i])
            P_by_year[index - 1].append(precipitation[i])
            Q_by_year[index - 1].append(runoff[i])
            if month < 5:
                Pcat[index - 1].append("winter")
                Qcat[index - 1].append("winter")
            else:
                Pcat[index - 1].append("summer")
                Qcat[index - 1].append("summer")
            if temperature[i] <= 0:
                Precip_cat[index - 1].append('snow')
            else:
                Precip_cat[index - 1].append('rain')
        else:
            date_daily_by_year[index].append(dates[i])
            P_by_year[index].append(precipitation[i])
            Q_by_year[index].append(runoff[i])
            Pcat[index].append("winter")
            Qcat[index].append("winter")
            if temperature[i] <= 0:
                Precip_cat[index].append('snow')
            else:
                Precip_cat[index].append('rain')
    return date_daily_by_year, P_by_year, Pcat, Q_by_year, Qcat, Precip_cat

# Remove the dates listed above and split isotope data into each year
# If the sampling period spans two seasons, add a date for the last day of the season to split up the fluxes by season
def SplitIsotopesByHydrologicYear(dates, intervals, years, no_count_date, precipitation, runoff):
    date_by_year = [[] for _ in years]
    precip_d_year = [[] for _ in years]
    Pdelcat_with_nan = [[] for _ in years]
    runoff_d_year = [[] for _ in years]
    Qdelcat_with_nan = [[] for _ in years]
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
                Pdelcat_with_nan[index - 1].append("winter")
                Pdelcat_with_nan[index - 1].append("summer")
                runoff_d_year[index - 1].append(runoff[i])
                runoff_d_year[index - 1].append(runoff[i])
                Qdelcat_with_nan[index - 1].append("winter")
                Qdelcat_with_nan[index - 1].append("summer")
                interval_by_year[index - 1].append(interval - days_in_month)
                interval_by_year[index - 1].append(days_in_month)
            else:
                date_by_year[index - 1].append(dates[i])
                precip_d_year[index - 1].append(precipitation[i])
                runoff_d_year[index - 1].append(runoff[i])
                interval_by_year[index - 1].append(interval)
                if month < 5:
                    Pdelcat_with_nan[index - 1].append("winter")
                    Qdelcat_with_nan[index - 1].append("winter")
                else:
                    Pdelcat_with_nan[index - 1].append("summer")
                    Qdelcat_with_nan[index - 1].append("summer")
        else:
            if month == 10 and days_in_month < interval:
                date_by_year[index-1].append(dt(int(year), 9, 30))
                date_by_year[index].append(dates[i])
                precip_d_year[index - 1].append(precipitation[i])
                precip_d_year[index].append(precipitation[i])
                Pdelcat_with_nan[index - 1].append("summer")
                Pdelcat_with_nan[index].append("winter")
                runoff_d_year[index - 1].append(runoff[i])
                runoff_d_year[index].append(runoff[i])
                Qdelcat_with_nan[index - 1].append("summer")
                Qdelcat_with_nan[index].append("winter")
                interval_by_year[index - 1].append(interval - days_in_month)
                interval_by_year[index].append(days_in_month)
            else:
                date_by_year[index].append(dates[i])
                precip_d_year[index].append(precipitation[i])
                Pdelcat_with_nan[index].append("winter")
                runoff_d_year[index].append(runoff[i])
                Qdelcat_with_nan[index].append("winter")
                interval_by_year[index].append(interval)
    return date_by_year, precip_d_year, Pdelcat_with_nan, runoff_d_year, Qdelcat_with_nan, interval_by_year