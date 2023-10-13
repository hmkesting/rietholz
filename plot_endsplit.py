from calc_endsplit import wtd_mean, endsplit
import statsmodels.api as sm
import scipy
import numpy as np
from textwrap import wrap
from matplotlib.pyplot import figure
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime as dt
import time
import statistics as stats

# Calculate confidence intervals
def calc_ci(t, s_err, n, x, x2, y2):
    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ci_upp = y2 + ci
    ci_low = y2 - ci
    return ci_low, ci_upp


# calculate ordinary least squares regression slope and intercept as well as standard errors (SE)
def ols_slope_int(x_vals, y_vals, plot=None):
    x, y = zip(*sorted(zip(x_vals, y_vals)))
    X = sm.add_constant(x)
    model_ols = sm.OLS(y, X).fit()
    int_se, slope_se = model_ols.bse
    intercept, slope = model_ols.params
    if plot != None:
        slope_pval = model_ols.pvalues[1]
        if slope_pval < 0.1:
            y_pred = model_ols.fittedvalues.tolist()
            df = model_ols.df_resid
            t_crit = abs(scipy.stats.t.ppf(q=0.025, df=df))
            x_val_ci = range(round(min(x_vals) - 5), round(max(x_vals) + 5), round((max(x_vals) - min(x_vals)) / 30))
            y_resid_sq = []
            for i in range(len(y)):
                y_resid_sq.append((y[i] - y_pred[i]) ** 2)
            s_err = np.sqrt(np.sum(y_resid_sq) / df)
            y_fitted = [0] * len(x_val_ci)
            for i in range(len(x_val_ci)):
                y_fitted[i] = model_ols.params[1] * x_val_ci[i] + model_ols.params[0]
            ci_low, ci_upp = calc_ci(t_crit, s_err, len(x), x, x_val_ci, y_fitted)
            plot.fill_between(x_val_ci, ci_upp, ci_low, alpha=.4, color='grey')
            plot.plot(x, y_pred, color='black', linewidth=2)
        plot.scatter(x, y, color='black')
    return [[slope, intercept], [slope_se, int_se]]


# Calculate time of year as a decimal
def to_year_fraction(date):
    def since_epoch(date):
        return time.mktime(date.timetuple())

    s = since_epoch
    year = date.year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year + 1, month=1, day=1)

    year_elapsed = s(date) - s(start_of_this_year)
    year_duration = s(start_of_next_year) - s(start_of_this_year)
    fraction = year_elapsed / year_duration

    return fraction


# Calculate seasonal and monthly weighted average precipitation isotope values and SE for Figure S3: Isotope Values
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
                sum_p_per_month[0] += (p_list[obs] * (days_in_month / interval_days))
                sum_p_per_month[11] += (p_list[obs] * ((interval_days - days_in_month) / interval_days))
            elif days_in_month < interval_days:
                p_months_lists[month - 1].append(p_list[obs] * (days_in_month / interval_days))
                p_months_lists[month - 2].append(p_list[obs] * ((interval_days - days_in_month) / interval_days))
                pdel_months_lists[month - 1].append(pdel_list[obs])
                pdel_months_lists[month - 2].append(pdel_list[obs])
                sum_p_per_month[month - 1] += (p_list[obs] * (days_in_month / interval_days))
                sum_p_per_month[month - 2] += (p_list[obs] * ((interval_days - days_in_month) / interval_days))
            else:
                p_months_lists[month - 1].append(p_list[obs])
                pdel_months_lists[month - 1].append(pdel_list[obs])
                sum_p_per_month[month - 1] += p_list[obs]

    # Calculate weighted means and errors for each month
    wtd_mean_per_month = [0] * 12
    s_error_per_month = [0] * 12
    for i in range(12):
        wtd_mean_per_month[i], s_error_per_month[i] = wtd_mean(pdel_months_lists[i], p_months_lists[i])

    # Calculate weighted means and errors for each season
    pdel_summer = []
    wts_summer = []
    for m in range((start_summer - 1), (start_winter - 1)):
        for i in range(len(p_months_lists[m])):
            pdel_summer.append(pdel_months_lists[m][i])
            wts_summer.append(p_months_lists[m][i])
    wtd_mean_summer, s_error_summer = wtd_mean(pdel_summer, wts_summer)

    pdel_winter = []
    wts_winter = []
    for m in range(0, (start_summer - 1)):
        for i in range(len(p_months_lists[m])):
            pdel_winter.append(pdel_months_lists[m][i])
            wts_winter.append(p_months_lists[m][i])
    for m in range((start_winter - 1), 12):
        for i in range(len(p_months_lists[m])):
            pdel_winter.append(pdel_months_lists[m][i])
            wts_winter.append(p_months_lists[m][i])
    wtd_mean_winter, s_error_winter = wtd_mean(pdel_winter, wts_winter)

    return wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month


# Calculate seasonal and monthly weighted average runoff isotope values and SE for Figure S3: Isotope Values
def calc_q(q_list_nan, qdel_list_nan, sample_dates, daily_dates):
    qdel_list = []
    q_list = []
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
        wtd_values.append(qdel_list[i] * q_list[i])

    wtd_mean_stream = sum(wtd_values) / sum(q_list)

    diff_stream_mean = []
    for i in range(len(qdel_list)):
        diff_stream_mean.append((qdel_list[i] - wtd_mean_stream) ** 2)

    left_num = []
    for i in range(len(q_list)):
        left_num.append(q_list[i] * diff_stream_mean[i])

    sqr_weights = []
    for i in range(len(q_list)):
        sqr_weights.append(q_list[i] ** 2)

    error_stream = math.sqrt((sum(left_num) / sum(q_list)) * ((sum(sqr_weights)) / ((sum(q_list)) ** 2 - sum(q_list))))
    return wtd_mean_stream, error_stream


def concatenate(data, year_indices, label=None):
    concatenated_list = []
    for y in year_indices:
        if label == None:
            for d in range(len(data[y])):
                concatenated_list.append(data[y][d])
        else:
            for d in range(len(data[y][label])):
                concatenated_list.append(data[y][label][d])
    return concatenated_list


# Conduct endsplitting over time periods of greater than 1 year
def multi_year_endsplit(iso_data, fluxes, pwt, qwt, year_indices):
    columns = ['Year', 'Ptot', 'P_s', 'P_s_se', 'P_w', 'P_w_se', 'Pdel_s', 'Pdel_w', 'Q', 'Qdel', 'ET', 'Qdel_s',
               'Qdel_w', 'Q_s', 'Q_w', 'ET_se', 'Q_s_se', 'Q_w_se']
    pdel = concatenate(iso_data, year_indices, label='Pdel')
    qdel = concatenate(iso_data, year_indices, label='Qdel')
    pwt = concatenate(pwt, year_indices)
    qwt = concatenate(qwt, year_indices)
    pdelcat = concatenate(iso_data, year_indices, label='Pdelcat')
    qdelcat = concatenate(iso_data, year_indices, label='Qdelcat')
    p = concatenate(fluxes, year_indices, label='P')
    q = concatenate(fluxes, year_indices, label='Q')
    pcat = concatenate(fluxes, year_indices, label='Pcat')
    qcat = concatenate(fluxes, year_indices, label='Qcat')
    l, table = endsplit(pdel, qdel, pwt, qwt, pdelcat, qdelcat, p, q, pcat, qcat)
    l[0] = len(year_indices)
    df = {columns[i]: l[i] for i in range(len(l))}
    return df, table


# Calculate the indices of the years which meet the cutoff '>= median' or '<= median'
def calc_year_indices(precip_df, years, season, cutoff):
    precip = []
    for i in range(len(precip_df['year'])):
        if precip_df['year'][i] in years:
            precip.append(precip_df[season][i])
    year_indices = []
    med = stats.median(precip)
    for i in range(len(precip)):
        if cutoff == '>= median':
            if precip[i] >= med:
                year_indices.append(i)
        elif cutoff == '<= median':
            if precip[i] <= med:
                year_indices.append(i)
        else:
            raise Exception('Choose splitting method')
    return year_indices


# Plot figure S3: Isotope Values
def plot_del_figure(q_all, stream_isotope, sampling_dates, date_daily, stream_isotope_upp, lysimeter_seepage,
                    isotope_lysimeter, precip_mm, precip_isotope, interval, iso_data_all, qwt_all, iso_data_upp,
                    qwt_upp, iso_data_lys, qwt_lys):
    start_summer = 4
    start_winter = 9
    wtd_mean_stream_all, error_stream_all = calc_q(q_all, stream_isotope, sampling_dates, date_daily)
    wtd_mean_stream_upper, error_stream_upper = calc_q(q_all, stream_isotope_upp, sampling_dates, date_daily)
    wtd_mean_stream_lys, error_stream_lys = calc_q(lysimeter_seepage, isotope_lysimeter, sampling_dates, date_daily)
    wtd_mean_winter, s_error_winter, wtd_mean_summer, s_error_summer, wtd_mean_per_month, s_error_per_month = \
        calc_precip(sampling_dates, precip_mm, precip_isotope, interval, start_summer, start_winter)
    wtd_mean_stream = [wtd_mean_stream_all, wtd_mean_stream_upper, wtd_mean_stream_lys]

    stream_label = [np.nan, 'Runoff average']
    colors = [np.nan, "blue", "green", "orange"]
    style = [np.nan, "solid", "dashed", "dashed"]
    date_all = concatenate(iso_data_all, range(len(iso_data_all)), label='Qdel_dates')
    date_upp = concatenate(iso_data_upp, range(len(iso_data_upp)), label='Qdel_dates')
    qdel_upp = concatenate(iso_data_upp, range(len(iso_data_upp)), label='Qdel')
    qwts_upp = concatenate(qwt_upp, range(len(iso_data_upp)))
    date_lys = concatenate(iso_data_lys, range(len(iso_data_lys)), label='Qdel_dates')
    qdate_upp = [to_year_fraction(x) * 11 for x in date_upp]

    s_error_summer_high = wtd_mean_summer + s_error_summer
    s_error_summer_low = wtd_mean_summer - s_error_summer
    s_error_winter_high = wtd_mean_winter + s_error_winter
    s_error_winter_low = wtd_mean_winter - s_error_winter
    letters_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(7, 3.5))
    plt.scatter(qdate_upp, qdel_upp, qwts_upp, color='blue', marker='.', label='Runoff measurements')
    for i in [1]:
        plt.plot((0, 11), (wtd_mean_stream[i], wtd_mean_stream[i]), color=colors[i], linewidth=2, linestyle=style[i],
                 label=stream_label[i])
    plt.plot((start_summer - 0.5, start_winter - 0.5), (wtd_mean_summer, wtd_mean_summer), color='yellow', linewidth=3,
             label='Summer precipitation')
    plt.plot((start_summer - 0.5, start_winter - 0.5), (s_error_summer_high, s_error_summer_high), color='yellow',
             linewidth=1)
    plt.plot((start_summer - 0.5, start_winter - 0.5), (s_error_summer_low, s_error_summer_low), color='yellow',
             linewidth=1)
    plt.plot((0, start_summer - 0.5), (wtd_mean_winter, wtd_mean_winter), color='grey', linewidth=3,
             label='Winter precipitation')
    plt.plot((0, start_summer - 0.5), (s_error_winter_high, s_error_winter_high), color='grey', linewidth=1)
    plt.plot((0, start_summer - 0.5), (s_error_winter_low, s_error_winter_low), color='grey', linewidth=1)
    plt.plot((start_winter - 0.5, 11), (wtd_mean_winter, wtd_mean_winter), color='grey', linewidth=3)
    plt.plot((start_winter - 0.5, 11), (s_error_winter_high, s_error_winter_high), color='grey', linewidth=1)
    plt.plot((start_winter - 0.5, 11), (s_error_winter_low, s_error_winter_low), color='grey', linewidth=1)
    plt.errorbar(letters_list, wtd_mean_per_month, yerr=s_error_per_month, fmt='.', color='black',
                 label='Monthly precipitation averages')
    plt.legend(ncol=2, bbox_to_anchor=(1.0, -0.22))
    #plt.title('Weighted δ$^{18}$O Values of Precipitation and Runoff')
    plt.xlabel('Month')
    plt.ylabel('δ$^{18}$O (‰)')
    plt.subplots_adjust(bottom=0.35)
    #plt.savefig(r'C:\Users\User\Documents\UNR\LASTSEMESTER!!\Project2\Pdel_wide.svg', dpi=500)
    plt.show()


# Generate a smooth transition from one y-value to another to create splitting diagrams
def y(lower, upper):
    y_range = upper - lower
    y = []
    multiplier = [.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.22, 0.35, 0.5, 0.65, 0.78, 0.88, 0.92, 0.95, 0.96, 0.97,
                  0.98, 0.99, 1, 1]
    for i in range(20):
        y.append(lower + y_range * multiplier[i])
    return y


# Create splitting diagrams, used plenty of extra key words in the function to manipulate label spacing
def diagram_in_out(ax, t, d, title, space = 80, Ps_ET_pct=0, Pw_ET_amt=0, Pw_ET_pct=0, Ps_Qs_amt=0, Ps_Qs_pct=0,
                   Pw_Qs_amt=0, Pw_Qs_pct=0, Ps_Qw_amt=0, Ps_Qw_pct=0):
    fsize = 28
    amt_place = 0.45
    x = list(np.arange(0, 1, 0.05))
    yr = d['Year']
    Pw = d['P_w'] / yr
    Pw_Qw = round(Pw * t['eta.winter']['winter'])
    Pw_Qs = round(Pw * t['eta.winter']['summer'])
    Ps = d['P_s'] / yr
    Ps_ET = round(Ps * t['eta.summer']['ET'])
    Ps_Qw = round(Ps * t['eta.summer']['winter'])
    Qw = round(Pw * t['eta.winter']['winter'] + Ps * t['eta.summer']['winter'])
    Qs = round(Pw * t['eta.winter']['summer'] + Ps * t['eta.summer']['summer'])
    Pw = round(Pw)
    Ps = round(Ps)
    base = y(0, 0)
    figure(figsize=(8, 8), dpi=350)
    ax.fill_between(x, base, y(Pw_Qw, Pw_Qw), color='blue', alpha=0.5)
    ax.fill_between(x, y(Pw_Qw, Qw + space), y(Pw_Qw + Pw_Qs, Qw + Pw_Qs + space), color='blue', alpha=0.5)
    if Pw - Pw_Qs - Pw_Qw >= 0:
        ax.fill_between(x, y(Pw_Qw + Pw_Qs, Qw + Qs + space * 2), y(Pw, Ps + Pw - Ps_ET + space * 2), color='blue',
                        alpha=0.5)
    else:
        ax.plot(x, y(Pw_Qw + Pw_Qs, Qw + Qs + space * 2), color='blue', linestyle='dashed')
    ax.fill_between(x, y(Pw + space * 2, Pw_Qw), y(Pw + Ps_Qw + space * 2, Qw), color='yellow', alpha=0.5)
    ax.fill_between(x, y(Pw + Ps_Qw + space * 2, Qw + Pw_Qs + space), y(Ps + Pw - Ps_ET + space * 2, Qw + Qs + space),
                    color='yellow', alpha=0.5)
    ax.fill_between(x, y(Ps + Pw - Ps_ET + space * 2, Ps + Pw - Ps_ET + space * 2), y(Ps + Pw + space * 2, Ps + Pw +
                                                                                space * 2), color='yellow', alpha=0.5)
    ax.axis("off")
    ax.text(0, Ps * 0.9 + Pw, "Summer P", fontsize=fsize)
    ax.text(0, Ps * 0.9 + Pw - 75, str(Ps) + " ± " + str(round(d['P_s_se'] / yr)) + " mm", fontsize=fsize)
    ax.text(0, 0.3 * Pw, "Winter P", fontsize=fsize)
    ax.text(0, 0.3 * Pw - 75, str(Pw) + " ± " + str(round(d['P_w_se'] / yr)) + " mm", fontsize=fsize)
    ax.text(amt_place, 0.9 * Ps + Pw + space, str(Ps_ET) + " ± " + str(round(Ps * t['eta.summer.se']['ET'])) + " mm",
            fontsize=fsize)
    ax.text(amt_place, Ps + Pw + space - Ps_ET + Pw_ET_amt, str(Pw - Pw_Qw - Pw_Qs) + " ± " + str(round(Pw *
                                                                    t['eta.winter.se']['ET'])) + " mm", fontsize=fsize)
    ax.text(amt_place, Qw + space + 0.8 * Qs + Ps_Qs_amt, str(Ps - Ps_ET - Ps_Qw) + " ± " + str(round(Ps *
                                                                t['eta.summer.se']['summer'])) + " mm", fontsize=fsize)
    ax.text(amt_place, Qw + space + 0.5 * Pw_Qs + Pw_Qs_amt, str(Pw_Qs) + " ± " + str(round(Pw *
                                                                t['eta.winter.se']['summer'])) + " mm", fontsize=fsize)
    ax.text(amt_place, Pw_Qw + 0.5 * Ps_Qw + Ps_Qw_amt, str(Ps_Qw) + " ± " + str(round(Ps * t['eta.summer.se']['winter'])) +
                                                                " mm", fontsize=fsize)
    ax.text(amt_place, 0.7 * Pw_Qw, str(Pw_Qw) + " ± " + str(round(Pw * t['eta.winter.se']['winter'])) + " mm", fontsize=fsize)
    ax.text(1.05, Pw + 0.8 * Ps + 80, str(round(t['f.summer']['ET'] * 100)) + " ± " +
            str(round(t['f.summer.se']['ET'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05, Pw + 0.8 * Ps, str(round(t['f.winter']['ET'] * 100)) + " ± " +
            str(round(t['f.winter.se']['ET'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05,  Qw + 0.6 * Qs, str(round(t['f.summer']['summer'] * 100)) + " ± " +
            str(round(t['f.summer.se']['summer'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05,  Qw + 0.6 * Qs - 80, str(round(t['f.winter']['summer'] * 100)) + " ± " +
            str(round(t['f.winter.se']['summer'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05, 0.5 * Qw - 80, str(round(t['f.summer']['winter'] * 100)) + " ± " +
            str(round(t['f.summer.se']['winter'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05, 0.5 * Qw - 160, str(round(t['f.winter']['winter'] * 100)) + " ± " +
            str(round(t['f.winter.se']['winter'] * 100)) + '%', fontsize=fsize)
    ax.text(1.05, Pw + 0.8 * Ps + 160, 'ET', fontsize=fsize)
    ax.text(1.05, Qw + 0.6 * Qs + 80, ('\n'.join(wrap('Summer Runoff', 6))), fontsize=fsize)
    ax.text(1.05, 0.5 * Qw, ('\n'.join(wrap('Winter Runoff', 6))), fontsize=fsize)
    ax.set_title(title + " (n=" + str(yr) + ")", fontsize=fsize, wrap=True)


# Plot panels of correlations between seasonal precipitation amounts and EMS variables, figures S7-S12
def plot_correlations(df, x_column, source):
    if x_column == 'P_s':
        other = 'P_w'
        x_label = 'Summer Precipitation (mm)'
        other_label = 'Winter Precipitation (mm)'
    elif x_column == 'P_w':
        other = 'P_s'
        x_label = 'Winter Precipitation (mm)'
        other_label = 'Summer Precipitation (mm)'
    fig, axs = plt.subplots(5, 2, figsize=(7, 10))
    #plt.suptitle('\n'.join(wrap('Annual Endsplitting Values for ' + source + ' Plotted Against ' + x_label + ', '
    #             'Trend lines shown if p < 0.1', 60)), fontsize=16)
    axs[0, 0].set_title('A. δ$^{18}$O Summer Precipitation', fontsize=14)
    ols_slope_int(df[x_column], df['Pdel_s'], plot=axs[0, 0])
    axs[0, 1].set_title('B. δ$^{18}$O Winter Precipitation', fontsize=14)
    ols_slope_int(df[x_column], df['Pdel_w'], plot=axs[0, 1])
    axs[1, 0].set_title('C. δ$^{18}$O Summer Runoff', fontsize=14)
    ols_slope_int(df[x_column], df['Qdel_s'], plot=axs[1, 0])
    axs[1, 1].set_title('D. δ$^{18}$O Winter Runoff', fontsize=14)
    ols_slope_int(df[x_column], df['Qdel_w'], plot=axs[1, 1])
    axs[2, 0].set_title('E. Summer Runoff (mm)', fontsize=14)
    ols_slope_int(df[x_column], df['Q_s'], plot=axs[2, 0])
    axs[2, 1].set_title('F. Winter Runoff (mm)', fontsize=14)
    ols_slope_int(df[x_column], df['Q_w'], plot=axs[2, 1])
    axs[3, 0].set_title('G. δ$^{18}$O Annual Runoff', fontsize=14)
    ols_slope_int(df[x_column], df['Qdel'], plot=axs[3, 0])
    axs[3, 1].set_title('H. Annual Runoff (mm)', fontsize=14)
    ols_slope_int(df[x_column], df['Q'], plot=axs[3, 1])
    axs[4, 0].set_title('I. ' + other_label, fontsize=14)
    ols_slope_int(df[x_column], df[other], plot=axs[4, 0])
    axs[4, 1].set_title('J. Evapotranspiration (mm)', fontsize=14)
    ols_slope_int(df[x_column], df['ET'], plot=axs[4, 1])
    axs[4, 0].set_xlabel(x_label, fontsize=12)
    axs[4, 1].set_xlabel(x_label, fontsize=12)
    fig.tight_layout()
    #fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\cor' + source + x_label + '.svg', dpi=500)
    plt.show()

