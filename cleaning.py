import math

# Sum precipitation and runoff values over each sampling period
def SumPrecipitationAndRunoff(years, dates_sampling, dates_daily, precipitation, runoff, Pdel, Qdel):
    Pwt = [[] for _ in years]
    Qwt = [[] for _ in years]
    sum_precip = 0
    sum_stream = 0

    for i in range(len(dates_daily)):
        for d in range(len(dates_daily[i])):
            if math.isnan(precipitation[i][d]):
                continue
            sum_precip += precipitation[i][d]
            sum_stream += runoff[i][d]
            if dates_daily[i][d] in dates_sampling[i]:
                index = dates_sampling[i].index(dates_daily[i][d])
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


