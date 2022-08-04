import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as scipy
from textwrap import wrap

# Calculate inverse error weighted regression. If the slope p-value > 0.1, return values to plot the line and the 95% CI
def calculate_wls(x_unsorted, y_unsorted, y_err_unsorted, x_bound=''):
    x, y, y_err = zip(*sorted(zip(x_unsorted, y_unsorted, y_err_unsorted)))
    w = []
    for i in y_err:
        w.append(1 / i ** 2)

    X = sm.add_constant(x)
    res_wls = sm.WLS(y, X, weights=w).fit()
    fitvals = res_wls.fittedvalues.tolist()
    slope_pval = res_wls.pvalues[1]
    slope = res_wls.params[1]
    if x_bound == 'P_s' or x_bound == 'P_w':
        x_val_ci = range(round(min(x)-5), round(max(x) + 5), 25)
    if x_bound == 'ratio':
        x_val_ci = np.linspace(min(x), max(x), 20)
    if slope_pval < 0.1:

        wtd_sum_x = []
        for i in range(len(w)):
            wtd_sum_x.append(w[i]*x[i])
        wtd_mean_x = sum(wtd_sum_x)/sum(w)

        int_se, slope_se = res_wls.bse
        df = res_wls.df_resid
        t_crit = abs(scipy.stats.t.ppf(q=0.025, df=df))
        ci_upp = [0] * len(x_val_ci)
        ci_low = [0] * len(x_val_ci)
        for i in range(len(x_val_ci)):
            ci_upp[i] = slope * x_val_ci[i] + res_wls.params[0] + t_crit * math.sqrt((abs(x_val_ci[i] -
                                                                            wtd_mean_x) * slope_se) ** 2 + int_se ** 2)
            ci_low[i] = slope * x_val_ci[i] + res_wls.params[0] - t_crit * math.sqrt((abs(x_val_ci[i] -
                                                                            wtd_mean_x) * slope_se) ** 2 + int_se ** 2)
    else:
        [fitvals, ci_low, ci_upp] = [[], [], []]
        [slope, slope_pval] = [0, 1]
    return {'xlabel':x, 'f':y, 'x_val_ci':x_val_ci, 'fitvals':fitvals,
            'ci_low':ci_low, 'ci_upp':ci_upp, 'slope': slope, 'slope p-val': slope_pval}

# The folowing function corrects ET values by correcting annual Ptot for undercatch (ET_lys_wts - ET_lys_water_bal),
# for years in which ET from lysimeter weights is known, and using the average undercatch otherwise
def undercatch_correction(watershed, undercatch):
    p_s = watershed['P_s'].tolist()
    p_w = watershed['P_w'].tolist()
    p_w_adj = copy.deepcopy(p_w)
    ptot = watershed['Ptot'].tolist()
    ptot_adj = copy.deepcopy(ptot)
    years = watershed['Year'].tolist()
    undercatch_years = list(undercatch['Year'])
    undercatch = undercatch['Undercatch (mm)']
    p_ratio_adj = []

    for i in range(len(years)):
        if years[i] in undercatch_years:
            ptot_adj[i] += undercatch[undercatch_years.index(years[i])]
            p_w_adj[i] += undercatch[undercatch_years.index(years[i])]
        else:
            ptot_adj[i] += sum(undercatch)/len(undercatch)
            p_w_adj[i] += sum(undercatch)/len(undercatch)
        p_ratio_adj.append(p_s[i] / p_w_adj[i])
    watershed_adj = copy.deepcopy(watershed)
    watershed_adj['Ptot'] = ptot_adj
    watershed_adj['P_w'] = p_w_adj
    watershed_adj['ratio'] = p_ratio_adj
    return watershed_adj

# Recalculate delta value of ET to recalculate the fraction of ET from summer precipitation
def calculate_fractions(watershed, et=None):
    p_s = watershed['P_s'].tolist()
    ptot = watershed['Ptot'].tolist()
    q = watershed['Q'].tolist()
    years = watershed['Year'].tolist()
    qdel = watershed['Qdel'].tolist()
    pdel_s = watershed['Pdel_s'].tolist()
    pdel_w = watershed['Pdel_w'].tolist()
    if et is None:
        et = watershed['ET'].tolist()
    else:
        et = [0] * len(years)
        for i in range(len(years)):
            et[i] = ptot[i] - q[i]

    f_et = [0]*len(years)
    f_et_se = [0]*len(years)
    f_ps = [0]*len(years)
    f_ps_se = [0]*len(years)
    for i in range(len(years)):
        f_et[i] = (p_s[i] - q[i] * (qdel[i] - pdel_w[i])/(pdel_s[i] - pdel_w[i]))/et[i]
        f_et_se[i] = abs(watershed['f_ET_se'][i] * f_et[i]/watershed['f_ET'][i])
        f_ps[i] = 1 - q[i]/p_s[i] * (qdel[i] - pdel_w[i])/(pdel_s[i] - pdel_w[i])
        f_ps_se[i] = abs(watershed['f_Ps_se'][i] * f_ps[i]/watershed['f_Ps'][i])
    new_watershed = copy.deepcopy(watershed)
    new_watershed['ET'] = et
    new_watershed['f_ET'] = f_et
    new_watershed['f_ET_se'] = f_et_se
    new_watershed['f_Ps'] = f_ps
    new_watershed['f_Ps_se'] = f_ps_se
    return new_watershed

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# Weighted Regression of Significant Slopes
def plot_panels(ci_all, ci_upper, ci_lys, x_label, title):
    print(title)
    count_f_et = {'TotalCount': {'+': 0, '-': 0, 'NA': 0},
     'ByWatershedCount': {'All RHB': {'+': 0, '-': 0, 'NA': 0}, 'Upper RHB': {'+': 0, '-': 0, 'NA': 0},
                          'Lysimeter': {'+': 0, '-': 0, 'NA': 0}},
     'ByMethodCount': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                       'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                       'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                       'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}},
     'ByWatershedandMethod': {
      'All RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                  'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                  'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                  'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}},
      'Upper RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                    'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                    'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                    'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}},
      'Lysimeter': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                    'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                    'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                    'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}}}}
    count_f_ps = {'TotalCount': {'+': 0, '-': 0, 'NA': 0},
    'ByWatershedCount': {'All RHB': {'+': 0, '-': 0, 'NA': 0}, 'Upper RHB': {'+': 0, '-': 0, 'NA': 0},
                         'Lysimeter': {'+': 0, '-': 0, 'NA': 0}},
    'ByMethodCount': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                      'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}},
    'ByWatershedandMethod': {'All RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                      'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}}, 'Upper RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                      'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}}, 'Lysimeter': {'No Lag': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 0},
                      'Mixed': {'+': 0, '-': 0, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 0},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 0}}}}

    fig, axs = plt.subplots(4, 3, figsize=(8.5, 10.5))
    fig.suptitle('\n'.join(wrap('Inverse Error Weighted Regressions with Significant (p<0.1) Slopes ' + title, 67)), y=0.99, x=0.4)
    axs[0, 0].set(ylabel=('\n'.join(wrap('Fraction of ET from Summer Precipitation (unitless)', 28))))
    axs[1, 0].set(ylabel=('\n'.join(wrap('Fraction of ET from Summer Precipitation (unitless)', 28))))
    axs[2, 0].set(ylabel=('\n'.join(wrap('Fraction of Summer Precipitation to ET (unitless)', 28))))
    axs[3, 0].set(xlabel=x_label, ylabel=('\n'.join(wrap('Fraction of Summer Precipitation to ET (unitless)', 28))))
    axs[3, 1].set(xlabel=x_label)
    axs[3, 2].set(xlabel=x_label)
    axes = [axs[0, 0], axs[0, 1], axs[0,2], axs[1, 0], axs[1, 1], axs[1, 2]]
    methods = ['No Lag', 'Lag 1', 'Lag 2', 'Mixed', 'Lag 1 Mean', 'Lag 2 Mean']
    method_labels = ['A. No Lag', 'B. Lag 1', 'C. Lag 2', 'D. Mixed', 'E. Lag 1 Mean', 'F. Lag 2 Mean']
    watersheds = [ci_all, ci_upper, ci_lys]
    colors = ['blue', 'orange', 'green']
    labels = ['All RHB', 'Upper RHB', 'Lysimeter']
    for m in range(len(methods)):
        x_values = [min(watersheds[0][methods[m]]['x_val_ci']), min(watersheds[1][methods[m]]['x_val_ci']),
                    min(watersheds[2][methods[m]]['x_val_ci']), max(watersheds[0][methods[m]]['x_val_ci']),
                    max(watersheds[1][methods[m]]['x_val_ci']), max(watersheds[2][methods[m]]['x_val_ci'])]
        axes[m].plot((min(x_values), max(x_values)), (0, 0), color='grey', linestyle='dashed')
        axes[m].plot((min(x_values), max(x_values)), (1, 1), color='grey', linestyle='dashed')
        for w in range(len(watersheds)):
            if watersheds[w][methods[m]]['f_ET_slope_pval'] < 0.1:
                axes[m].plot(watersheds[w][methods[m]]['xlabel'], watersheds[w][methods[m]]['f_ET_fitvals'], color=colors[w], linewidth=2,)
                axes[m].plot(watersheds[w][methods[m]]['x_val_ci'], watersheds[w][methods[m]]['f_ET_ci_low'], color=colors[w], linewidth=0.75)
                axes[m].plot(watersheds[w][methods[m]]['x_val_ci'], watersheds[w][methods[m]]['f_ET_ci_upp'], color=colors[w], linewidth=0.75)
                if watersheds[w][methods[m]]['f_ET_slope'] > 0:
                    count_f_et['TotalCount']['+'] += 1
                    count_f_et['ByWatershedCount'][labels[w]]['+'] += 1
                    count_f_et['ByMethodCount'][methods[m]]['+'] += 1
                    count_f_et['ByWatershedandMethod'][labels[w]][methods[m]]['+'] += 1
                if watersheds[w][methods[m]]['f_ET_slope'] < 0:
                    count_f_et['TotalCount']['-'] += 1
                    count_f_et['ByWatershedCount'][labels[w]]['-'] += 1
                    count_f_et['ByMethodCount'][methods[m]]['-'] += 1
                    count_f_et['ByWatershedandMethod'][labels[w]][methods[m]]['-'] += 1
            else:
                count_f_et['TotalCount']['NA'] += 1
                count_f_et['ByWatershedCount'][labels[w]]['NA'] += 1
                count_f_et['ByMethodCount'][methods[m]]['NA'] += 1
                count_f_et['ByWatershedandMethod'][labels[w]][methods[m]]['NA'] += 1
        axes[m].set_title(method_labels[m])
    print('f_ET')
    for m in range(len(methods)):
        for w in range(len(watersheds)):
            for i in range(len(watersheds[w][methods[m]]['f_ET'])):
                if watersheds[w][methods[m]]['f_ET'][i] > 5 or watersheds[w][methods[m]]['f_ET'][i] < -5:
                    if x_label == "Winter Precipitation (mm)" or x_label == "Summer Precipitation (mm)":
                        print(labels[w], method_labels[m], 'points not shown at (', round_half_up(watersheds[w][methods[m]]['xlabel'][i], 0), round_half_up(watersheds[w][methods[m]]['f_ET'][i], 1), ')')
                    else:
                        print(labels[w], method_labels[m], 'points not shown at (',
                              round_half_up(watersheds[w][methods[m]]['xlabel'][i], 2),
                              round_half_up(watersheds[w][methods[m]]['f_ET'][i], 1), ')')
            axes[m].scatter(watersheds[w][methods[m]]['xlabel'], watersheds[w][methods[m]]['f_ET'], color=colors[w], marker='.', label=labels[w])
            axes[m].set_ylim([-5, 5])

    axes = axs[2, 0], axs[2, 1], axs[2, 2], axs[3, 0], axs[3, 1], axs[3, 2]
    methods = ['No Lag', 'Lag 1', 'Lag 2', 'Mixed', 'Lag 1 Mean', 'Lag 2 Mean']
    method_labels = ['G. No Lag', 'H. Lag 1', 'I. Lag 2', 'J. Mixed', 'K. Lag 1 Mean', 'L. Lag 2 Mean']
    for m in range(len(methods)):
        x_values = [min(watersheds[0][methods[m]]['x_val_ci']), min(watersheds[1][methods[m]]['x_val_ci']),
                    min(watersheds[2][methods[m]]['x_val_ci']), max(watersheds[0][methods[m]]['x_val_ci']),
                    max(watersheds[1][methods[m]]['x_val_ci']), max(watersheds[2][methods[m]]['x_val_ci'])]
        axes[m].plot((min(x_values), max(x_values)), (0, 0), color='grey', linestyle='dashed')
        axes[m].plot((min(x_values), max(x_values)), (1, 1), color='grey', linestyle='dashed')
        for w in range(len(watersheds)):
            if watersheds[w][methods[m]]['f_Ps_slope_pval'] < 0.1:
                axes[m].plot(watersheds[w][methods[m]]['xlabel'], watersheds[w][methods[m]]['f_Ps_fitvals'], color=colors[w], linewidth=2)
                axes[m].plot(watersheds[w][methods[m]]['x_val_ci'], watersheds[w][methods[m]]['f_Ps_ci_low'], color=colors[w], linewidth=0.75)
                axes[m].plot(watersheds[w][methods[m]]['x_val_ci'], watersheds[w][methods[m]]['f_Ps_ci_upp'], color=colors[w], linewidth=0.75)
                if watersheds[w][methods[m]]['f_Ps_slope'] > 0:
                    count_f_ps['TotalCount']['+'] += 1
                    count_f_ps['ByWatershedCount'][labels[w]]['+'] += 1
                    count_f_ps['ByMethodCount'][methods[m]]['+'] += 1
                    count_f_ps['ByWatershedandMethod'][labels[w]][methods[m]]['+'] += 1
                if watersheds[w][methods[m]]['f_Ps_slope'] < 0:
                    count_f_ps['TotalCount']['-'] += 1
                    count_f_ps['ByWatershedCount'][labels[w]]['-'] += 1
                    count_f_ps['ByMethodCount'][methods[m]]['-'] += 1
                    count_f_ps['ByWatershedandMethod'][labels[w]][methods[m]]['-'] += 1
            else:
                count_f_ps['TotalCount']['NA'] += 1
                count_f_ps['ByWatershedCount'][labels[w]]['NA'] += 1
                count_f_ps['ByMethodCount'][methods[m]]['NA'] += 1
                count_f_ps['ByWatershedandMethod'][labels[w]][methods[m]]['NA'] += 1
        axes[m].set_title(method_labels[m])
    print('f_Ps')
    for m in range(len(methods)):
        for w in range(len(watersheds)):
            for i in range(len(watersheds[w][methods[m]]['f_ET'])):
                if watersheds[w][methods[m]]['f_Ps'][i] > 5 or watersheds[w][methods[m]]['f_Ps'][i] < -5:
                    if x_label == "Winter Precipitation (mm)" or x_label == "Summer Precipitation (mm)":
                        print(labels[w], method_labels[m], 'points not shown at (',
                              round_half_up(watersheds[w][methods[m]]['xlabel'][i], 0),
                              round_half_up(watersheds[w][methods[m]]['f_Ps'][i], 1), ')')
                    else:
                        print(labels[w], method_labels[m], 'points not shown at (',
                              round_half_up(watersheds[w][methods[m]]['xlabel'][i], 2),
                              round_half_up(watersheds[w][methods[m]]['f_Ps'][i], 1), ')')
            axes[m].scatter(watersheds[w][methods[m]]['xlabel'], watersheds[w][methods[m]]['f_Ps'], color=colors[w], marker='.', label=labels[w])
            axes[m].set_ylim([-5, 5])

    fig.tight_layout(pad=1)
    plt.legend(bbox_to_anchor=(0.7, 5.6), labelspacing=0.2, frameon=False)
    return count_f_et, count_f_ps

# Figure showing annual ET amount by catchment and year
def plot_et_amounts(x, df_no_lag_all, df_no_lag_upper, df_no_lag_lys, evapotranspiration):

    et_all = [0] * len(x)
    et_upper = [0] * len(x)
    et_lys = [0] * len(x)
    et_wts = [0] * len(x)
    years_upper = copy.deepcopy(x)
    years_lys = copy.deepcopy(x)
    years_wts = copy.deepcopy(x)

    for i in range(len(x)):
        years_upper[i] += 0.15
        years_lys[i] += 0.3
        years_wts[i] += 0.45
        if float(x[i]) in df_no_lag_all['Year'].tolist():
            et_all[i] = df_no_lag_all['ET'][df_no_lag_all['Year'].tolist().index(x[i])]
        if x[i] in df_no_lag_upper['Year'].tolist():
            et_upper[i] = df_no_lag_upper['ET'][df_no_lag_upper['Year'].tolist().index(x[i])]
        if x[i] in df_no_lag_lys['Year'].tolist():
            et_lys[i] = df_no_lag_lys['ET'][df_no_lag_lys['Year'].tolist().index(x[i])]
        if x[i] in evapotranspiration['Year'].tolist():
            et_wts[i] = evapotranspiration['annual_ET'][evapotranspiration['Year'].tolist().index(x[i])]

    plt.figure(figsize=(8, 4.5))
    plt.title('Annual Evapotranspiration Values')
    plt.xlabel('Year')
    plt.xticks([1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012])
    plt.ylabel('Evapotranspiration (mm)')

    data = {"All": et_all, "Upper": et_upper, "Lysimeter discharge": et_lys, "Lysimeter weights": et_wts}
    plt.bar(x, data['All'], color='purple', width=0.15, label='All Water Balance')
    plt.bar(years_upper, data['Upper'], color='b', width=0.15, label='Upper Water Balance')
    plt.bar(years_lys, data['Lysimeter discharge'], color='r', width=0.15, label='Lysimeter Water Balance')
    plt.bar(years_wts, data['Lysimeter weights'], color='black', width=0.15, label='Lysimeter Weights')

    plt.legend(bbox_to_anchor=(0.56, 0.28))
    #plt.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\et_amounts.eps', dpi=400)
    plt.show()
    return data

def calculate_avg_et(watershed):
    ptot = watershed['Ptot'].tolist()
    q = watershed['Q'].tolist()
    avg_et = (sum(ptot)-sum(q))/len(ptot)
    avg_et_list = []
    q_rescaled = []
    for i in range(len(ptot)):
        avg_et_list.append(avg_et)
        q_rescaled.append(ptot[i] - avg_et)
    watershed_updated = copy.deepcopy(watershed)
    watershed_updated['ET'] = avg_et_list
    watershed_updated['Q'] = q_rescaled
    return watershed_updated

def calculate_scaled_et(watershed, lysimeter_weights):
    watershed_years = watershed['Year'].tolist()
    ptot = watershed['Ptot'].tolist()
    q = watershed['Q'].tolist()
    sum_watershed_et = sum(ptot)-sum(q)
    lysimeter_years_nan = lysimeter_weights['Year'].tolist()
    lysimeter_et_nan = lysimeter_weights['annual_ET'].tolist()
    lysimeter_years = []
    lysimeter_et = []
    for i in range(len(lysimeter_years_nan)):
        if np.isnan(lysimeter_et_nan[i]):
            continue
        else:
            lysimeter_years.append(lysimeter_years_nan[i])
            lysimeter_et.append(lysimeter_et_nan[i])
    new_et = []
    new_q = []
    fill_lysimeter_et = []
    for i in range(len(watershed_years)):
        if watershed_years[i] in lysimeter_years:
            fill_lysimeter_et.append(lysimeter_et[lysimeter_years.index(watershed_years[i])])
        else:
            fill_lysimeter_et.append(sum(lysimeter_et)/len(lysimeter_et))
    for i in range(len(watershed_years)):
        new_et.append(sum_watershed_et * (fill_lysimeter_et[i]/sum(fill_lysimeter_et)))
        new_q.append(ptot[i] - new_et[i])
    watershed_updated = copy.deepcopy(watershed)
    watershed_updated['ET'] = new_et
    watershed_updated['Q'] = new_q
    return watershed_updated

