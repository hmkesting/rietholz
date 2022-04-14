import math

# Sum precipitation and runoff values over each sampling period
def SumPrecipitationAndRunoff(dates_sampling, fluxes, Pdel, Qdel):
    Pwt = [[] for _ in fluxes]
    Qwt = [[] for _ in fluxes]
    sum_precip = 0
    sum_stream = 0

    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            if math.isnan(fluxes[i]['P'][d]):
                continue
            sum_precip += fluxes[i]['P'][d]
            sum_stream += fluxes[i]['Q'][d]
            if fluxes[i]['dates'][d] in dates_sampling[i]:
                index = dates_sampling[i].index(fluxes[i]['dates'][d])
                if math.isnan(Pdel[i][index]) and math.isnan(Qdel[i][index]):
                    sum_precip = 0
                    sum_stream = 0
                    continue
                elif math.isnan(Pdel[i][index]) and not math.isnan(Qdel[i][index]):
                    Qwt[i].append(sum_stream)
                elif not math.isnan(Pdel[i][index]) and math.isnan(Qdel[i][index]):
                    Pwt[i].append(sum_precip)
                else:
                    Pwt[i].append(sum_precip)
                    Qwt[i].append(sum_stream)
                sum_precip = 0
                sum_stream = 0
    return Pwt, Qwt

# Remove NaN (missing data) and associated categorical variable from isotope data
def RemoveNanSamples(years, samples, categorical):
    isotopes = [[] for _ in years]
    category = [[] for _ in years]
    for i in range(len(years)):
        for d in range(len(samples[i])):
            if math.isnan(samples[i][d]):
                continue
            else:
                isotopes[i].append(samples[i][d])
                category[i].append(categorical[i][d])
    return isotopes, category

# Identify which years need to be removed due to a lack of runoff data (will not work for missing precipitation data)
def IdentifyMissingData(fluxes, interval):
    aux_list_precip = [[] for _ in fluxes]
    count_list_precip = [[] for _ in fluxes]
    no_data_dates_precip = [[] for _ in fluxes]
    aux_list_runoff = [[] for _ in fluxes]
    count_list_runoff = [[] for _ in fluxes]
    no_data_dates_runoff = [[] for _ in fluxes]
    remove_years = []
    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            month_year = fluxes[i]['dates'][d].strftime("%b '%y")
            if math.isnan(fluxes[i]['P'][d]) or math.isnan(fluxes[i]['Q'][d]):
                if math.isnan(fluxes[i]['P'][d]) and not math.isnan(fluxes[i]['Q'][d]):
                    if month_year not in aux_list_precip[i]:
                        aux_list_precip[i].append(month_year)
                        no_data_dates_precip[i].append(fluxes[i]['dates'][d])
                        count_list_precip[i].append(interval[i][d])
                    else:
                        index = aux_list_precip[i].index(month_year)
                        no_data_dates_precip[i].append(fluxes[i]['dates'][d])
                        count_list_precip[i][index] += interval[i][d]
                elif math.isnan(fluxes[i]['P'][d]) and math.isnan(fluxes[i]['Q'][d]):
                    if month_year not in aux_list_precip[i]:
                        aux_list_precip[i].append(month_year)
                        no_data_dates_precip[i].append(fluxes[i]['dates'][d])
                        count_list_precip[i].append(interval[i][d])
                    else:
                        index = aux_list_precip[i].index(month_year)
                        no_data_dates_precip[i].append(fluxes[i]['dates'][d])
                        count_list_precip[i][index] += interval[i][d]
                    if month_year not in aux_list_runoff[i]:
                        aux_list_runoff[i].append(month_year)
                        no_data_dates_runoff[i].append(fluxes[i]['dates'][d])
                        count_list_runoff[i].append(interval[i][d])
                    else:
                        index = aux_list_runoff[i].index(month_year)
                        no_data_dates_runoff[i].append(fluxes[i]['dates'][d])
                        count_list_runoff[i][index] += interval[i][d]
                elif not math.isnan(fluxes[i]['P'][d]) and math.isnan(fluxes[i]['Q'][d]):
                    if month_year not in aux_list_runoff[i]:
                        aux_list_runoff[i].append(month_year)
                        no_data_dates_runoff[i].append(fluxes[i]['dates'][d])
                        count_list_runoff[i].append(interval[i][d])
                    else:
                        index = aux_list_runoff[i].index(month_year)
                        no_data_dates_runoff[i].append(fluxes[i]['dates'][d])
                        count_list_runoff[i][index] += interval[i][d]
    #The commented lines below print the number of days per year missing precipitation and runoff data as well as which months are missing data
    #for i in range(len(aux_list_precip)):
        #if aux_list_precip[i]:
            #print(sum(count_list_precip[i]), 'days of precipitation values missing in', aux_list_precip[i])
    for i in range(len(aux_list_runoff)):
        if aux_list_runoff[i]:
            #print(sum(count_list_runoff[i]), 'days of runoff values missing in', aux_list_runoff[i])
            if sum(count_list_runoff[i]) > 5:
                remove_years.append(fluxes[i]['Year'])
    return remove_years

