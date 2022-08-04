import math

# Sum precipitation and runoff values over each sampling period
def sum_precipitation_and_runoff(dates_sampling, fluxes, pdel, qdel):
    pwt = [[] for _ in fluxes]
    qwt = [[] for _ in fluxes]
    sum_precip = 0

    for i in range(len(fluxes)):
        for d in range(len(fluxes[i]['dates'])):
            if math.isnan(fluxes[i]['P'][d]):
                continue
            sum_precip += fluxes[i]['P'][d]
            daily_stream = fluxes[i]['Q'][d]
            if fluxes[i]['dates'][d] in dates_sampling[i]:
                index = dates_sampling[i].index(fluxes[i]['dates'][d])
                if math.isnan(pdel[i][index]) and math.isnan(qdel[i][index]):
                    sum_precip = 0
                    continue
                elif math.isnan(pdel[i][index]) and not math.isnan(qdel[i][index]):
                    qwt[i].append(daily_stream)
                elif not math.isnan(pdel[i][index]) and math.isnan(qdel[i][index]):
                    pwt[i].append(sum_precip)
                else:
                    pwt[i].append(sum_precip)
                    qwt[i].append(daily_stream)
                sum_precip = 0
    return pwt, qwt

# Remove NaN (missing data) and associated categorical variable from isotope data
def remove_nan_samples(years, samples, categorical):
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

