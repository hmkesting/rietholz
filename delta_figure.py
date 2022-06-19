import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime as dt
from calculations import wtd_mean
import time

# Calculate time of year as a decimal
def to_year_fraction(date):
    def since_epoch(date):
        return time.mktime(date.timetuple())
    s = since_epoch

    year = date.year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year+1, month=1, day=1)

    year_elapsed = s(date) - s(start_of_this_year)
    year_duration = s(start_of_next_year) - s(start_of_this_year)
    fraction = year_elapsed/year_duration

    return fraction
    
def calc_precip(list_of_dates, p_list, pdel_list, days_interval, start_summer, start_winter):
    p_months_lists = [[] for _ in range(12)]
    pdel_months_lists = [[] for _ in range(12)]
    sum_p_per_month = [0] * 12
    days_interval = days_interval.fillna(0).to_list()
    for obs, date in enumerate(list_of_dates):
        if obs != 0 and not pd.isna(pdel_list[obs]):
            mmddyy = date.split('/')
            month = int(mmddyy[0])
            days_in_month = int(mmddyy[1])
            interval_days = int(days_interval[obs])
            if days_in_month < interval_days and month == 1:
                p_months_lists[0].append(p_list[obs] * (days_in_month / interval_days))
                p_months_lists[11].append(p_list[obs] * ((interval_days - days_in_month) / interval_days))
                pdel_months_lists[0].append(pdel_list[obs])
                pdel_months_lists[11].append(pdel_list[obs])
                sum_p_per_month[0] += (p_list[obs]*(days_in_month / interval_days))
                sum_p_per_month[11] += (p_list[obs] * ((interval_days - days_in_month) / interval_days))
            elif days_in_month < interval_days:
                p_months_lists[month-1].append(p_list[obs] * (days_in_month / interval_days))
                p_months_lists[month-2].append(p_list[obs] * ((interval_days - days_in_month) / interval_days))
                pdel_months_lists[month-1].append(pdel_list[obs])
                pdel_months_lists[month-2].append(pdel_list[obs])
                sum_p_per_month[month-1] += (p_list[obs] * (days_in_month / interval_days))
                sum_p_per_month[month-2] += (p_list[obs] * ((interval_days - days_in_month) / interval_days))
            else:
                p_months_lists[month-1].append(p_list[obs])
                pdel_months_lists[month - 1].append(pdel_list[obs])
                sum_p_per_month[month - 1] += p_list[obs]

    # Calculate weighted means and errors for each month
    wtd_mean_per_month = [0]*12
    s_error_per_month = [0]*12
    for i in range(12):
        wtd_mean_per_month[i], s_error_per_month[i] = wtd_mean(pdel_months_lists[i], p_months_lists[i])

    # Calculate weighted means and errors for each season
    pdel_summer = []
    wts_summer = []
    for m in range((start_summer-1), (start_winter-1)):
        for i in range(len(p_months_lists[m])):
            pdel_summer.append(pdel_months_lists[m][i])
            wts_summer.append(p_months_lists[m][i])
    wtd_mean_summer, s_error_summer = wtd_mean(pdel_summer, wts_summer)

    pdel_winter = []
    wts_winter = []
    for m in range(0, (start_summer-1)):
        for i in range(len(p_months_lists[m])):
            pdel_winter.append(pdel_months_lists[m][i])
            wts_winter.append(p_months_lists[m][i])
    for m in range((start_winter-1), 12):
        for i in range(len(p_months_lists[m])):
            pdel_winter.append(pdel_months_lists[m][i])
            wts_winter.append(p_months_lists[m][i])
    wtd_mean_winter, s_error_winter = wtd_mean(pdel_winter, wts_winter)

    return wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month

def calc_q(q_list_nan, qdel_list_nan, sample_dates, daily_dates):
    qdel_list=[]
    q_list=[]
    sample_date_list = []

    for i in range(len(qdel_list_nan)):
        if pd.isna(qdel_list_nan[i]):
            continue
        else:
            qdel_list.append(qdel_list_nan[i])
            sample_date_list.append(dt.strptime(sample_dates[i], "%m/%d/%Y"))

    d = []
    for i in range(len(daily_dates)):
        d.append(dt.strptime(daily_dates[i], "%m/%d/%Y"))

    for i in range(len(d)):
        if d[i] in sample_date_list and not pd.isna(q_list_nan[i]):
            q_list.append(q_list_nan[i])
    wtd_values = []
    for i in range(len(q_list)):
        wtd_values.append(qdel_list[i]*q_list[i])

    wtd_mean_stream = sum(wtd_values)/sum(q_list)

    diff_stream_mean = []
    for i in range(len(qdel_list)):
        diff_stream_mean.append((qdel_list[i]-wtd_mean_stream)**2)

    left_num = []
    for i in range(len(q_list)):
        left_num.append(q_list[i]*diff_stream_mean[i])

    sqr_weights = []
    for i in range(len(q_list)):
        sqr_weights.append(q_list[i]**2)

    error_stream = math.sqrt((sum(left_num)/sum(q_list))*((sum(sqr_weights))/((sum(q_list))**2-sum(q_list))))
    return wtd_mean_stream, error_stream

def plot_del_figure(start_summer, start_winter, wtd_mean_summer, s_error_summer, wtd_mean_winter, s_error_winter,
                   wtd_mean_per_month, s_error_per_month, wtd_mean_stream, stream_label, colors, date_all, date_upper,
                   date_lys, stream_isotope, stream_isotope_upper, isotope_lysimeter_seepage, all_pt_size,
                   upper_pt_size, lys_pt_size):
    s_error_summer_high = wtd_mean_summer + s_error_summer
    s_error_summer_low = wtd_mean_summer - s_error_summer
    s_error_winter_high = wtd_mean_winter + s_error_winter
    s_error_winter_low = wtd_mean_winter - s_error_winter
    letters_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(7.5, 3.5))

    plt.scatter(date_all, stream_isotope, all_pt_size, color='blue', marker='.', label='All')
    plt.scatter(date_upper, stream_isotope_upper, upper_pt_size, color='orange', marker='.', label='Upper')
    plt.scatter(date_lys, isotope_lysimeter_seepage, lys_pt_size, color='green', marker='.', label='Lysimeter')
    plt.plot((start_summer-0.5, start_winter-0.5), (wtd_mean_summer, wtd_mean_summer), color='yellow', linewidth=3, label='Summer precipitation')
    plt.plot((start_summer-0.5, start_winter-0.5), (s_error_summer_high, s_error_summer_high), color='yellow', linewidth=1)
    plt.plot((start_summer-0.5, start_winter-0.5), (s_error_summer_low, s_error_summer_low), color='yellow', linewidth=1)
    plt.plot((0, start_summer-0.5), (wtd_mean_winter, wtd_mean_winter), color='grey', linewidth=3, label='Winter precipitation')
    plt.plot((0, start_summer-0.5), (s_error_winter_high, s_error_winter_high), color='grey', linewidth=1)
    plt.plot((0, start_summer-0.5), (s_error_winter_low, s_error_winter_low), color='grey', linewidth=1)
    plt.plot((start_winter-0.5, 11), (wtd_mean_winter, wtd_mean_winter), color='grey', linewidth=3)
    plt.plot((start_winter-0.5, 11), (s_error_winter_high, s_error_winter_high), color='grey', linewidth=1)
    plt.plot((start_winter-0.5, 11), (s_error_winter_low, s_error_winter_low), color='grey', linewidth=1)
    for i in [0,2]:
        plt.plot((0, 11), (wtd_mean_stream[i], wtd_mean_stream[i]), color=colors[i], linewidth=1, label=stream_label[i])
    plt.errorbar(letters_list, wtd_mean_per_month, yerr=s_error_per_month, fmt='.', color='black', label='Monthly averages')
    plt.legend(bbox_to_anchor=(1.01, 0.95))
    plt.title('Weighted δ$^{18}$O Values of Precipitation and Runoff')
    plt.xlabel('Month')
    plt.ylabel('δ$^{18}$O (‰)')
    plt.tight_layout()
    plt.show()