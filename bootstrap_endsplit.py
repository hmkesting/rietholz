from calc_endsplit import endsplit
from plot_endsplit import ols_slope_int
import numpy as np
import random
from random import choices
import statistics
from textwrap import wrap
import matplotlib.pyplot as plt
import pandas as pd


# Use the numerical solution approach to calculate partitioning fractions by substituting each variable in the EMM
# equations with a linear regression between that variable and the x-axis (seasonal precipitation amount)
def calculate_yvals(x_column, x_axis, Pdel_s, Pdel_w, P_s, P_w, Q_s, Q_w, Qdel_s, Qdel_w, f_Qs_from_Ps, f_Qw_from_Ps,
                    f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw):
    if x_column == 'P_s':
        X = P_s
    elif x_column == 'P_w':
        X = P_w
    pdels, pdels_se = ols_slope_int(X, Pdel_s)
    pdelw, pdelw_se = ols_slope_int(X, Pdel_w)
    qs, qs_se = ols_slope_int(X, Q_s)
    qw, qw_se = ols_slope_int(X, Q_w)
    qdels, qdels_se = ols_slope_int(X, Qdel_s)
    qdelw, qdelw_se = ols_slope_int(X, Qdel_w)

    def regression(mb):
        result = mb[0] * x_axis[i] + mb[1]
        return result

    def calc_f_q_from_ps(qdel, pdelw, pdels):
        return (qdel - pdelw) / (pdels - pdelw)

    def calc_f_ps_to_q(f_q_from_ps, q, ps):
        return f_q_from_ps * q / ps

    if x_column == 'P_s':
        pw, pw_se = ols_slope_int(X, P_w)
        for i in range(len(x_axis)):
            f_Qs_Ps = calc_f_q_from_ps(regression(qdels), regression(pdelw), regression(pdels))
            f_Qw_Ps = calc_f_q_from_ps(regression(qdelw), regression(pdelw), regression(pdels))
            f_Ps_Qs = calc_f_ps_to_q(f_Qs_Ps, regression(qs), x_axis[i])
            f_Ps_Qw = calc_f_ps_to_q(f_Qw_Ps, regression(qw), x_axis[i])
            f_Pw_Qs = calc_f_ps_to_q((1 - f_Qs_Ps), regression(qs), regression(pw))
            f_Pw_Qw = calc_f_ps_to_q((1 - f_Qw_Ps), regression(qw), regression(pw))
            f_Qs_from_Ps[i].append(f_Qs_Ps)
            f_Qw_from_Ps[i].append(f_Qw_Ps)
            f_Ps_to_Qs[i].append(f_Ps_Qs)
            f_Ps_to_Qw[i].append(f_Ps_Qw)
            f_Pw_to_Qs[i].append(f_Pw_Qs)
            f_Pw_to_Qw[i].append(f_Pw_Qw)
            mm_Ps_to_Qs[i].append(f_Ps_Qs * x_axis[i])
            mm_Ps_to_Qw[i].append(f_Ps_Qw * x_axis[i])
            mm_Pw_to_Qs[i].append(f_Pw_Qs * regression(pw))
            mm_Pw_to_Qw[i].append(f_Pw_Qw * regression(pw))
    if x_column == 'P_w':
        ps, ps_se = ols_slope_int(X, P_s)
        for i in range(len(x_axis)):
            f_Qs_Ps = calc_f_q_from_ps(regression(qdels), regression(pdelw), regression(pdels))
            f_Qw_Ps = calc_f_q_from_ps(regression(qdelw), regression(pdelw), regression(pdels))
            f_Ps_Qs = calc_f_ps_to_q(f_Qs_Ps, regression(qs), regression(ps))
            f_Ps_Qw = calc_f_ps_to_q(f_Qw_Ps, regression(qw), regression(ps))
            f_Pw_Qs = calc_f_ps_to_q((1 - f_Qs_Ps), regression(qs), x_axis[i])
            f_Pw_Qw = calc_f_ps_to_q((1 - f_Qw_Ps), regression(qw), x_axis[i])
            f_Qs_from_Ps[i].append(f_Qs_Ps)
            f_Qw_from_Ps[i].append(f_Qw_Ps)
            f_Ps_to_Qs[i].append(f_Ps_Qs)
            f_Ps_to_Qw[i].append(f_Ps_Qw)
            f_Pw_to_Qs[i].append(f_Pw_Qs)
            f_Pw_to_Qw[i].append(f_Pw_Qw)
            mm_Ps_to_Qs[i].append(f_Ps_Qs * regression(ps))
            mm_Ps_to_Qw[i].append(f_Ps_Qw * regression(ps))
            mm_Pw_to_Qs[i].append(f_Pw_Qs * x_axis[i])
            mm_Pw_to_Qw[i].append(f_Pw_Qw * x_axis[i])
    return f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
           mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw


# Iterate through random seed 1 to 3000 to select samples of the years, then apply the samples using the numerical
# solution approach and record the resulting partitioning fractions for each iteration
def bootstrapping_numerical(df, x_column, X_axis):
    # y values used to calculate percentiles and medians for each segment of the numerical solution plot
    f_Qs_from_Ps = [[] for _ in X_axis]
    f_Ps_to_Qs = [[] for _ in X_axis]
    f_Qw_from_Ps = [[] for _ in X_axis]
    f_Ps_to_Qw = [[] for _ in X_axis]
    f_Pw_to_Qs = [[] for _ in X_axis]
    f_Pw_to_Qw = [[] for _ in X_axis]
    mm_Ps_to_Qs = [[] for _ in X_axis]
    mm_Ps_to_Qw = [[] for _ in X_axis]
    mm_Pw_to_Qs = [[] for _ in X_axis]
    mm_Pw_to_Qw = [[] for _ in X_axis]
    for n in range(3000):
        Pdel_s = []
        Pdel_w = []
        P_w = []
        P_s = []
        Qdel_s = []
        Qdel_w = []
        Q_s = []
        Q_w = []
        length = len(df[x_column])
        random.seed(n)
        sample_indices = choices(range(len(df[x_column])), k=length)
        for i in sample_indices:
            Pdel_s.append(df['Pdel_s'][i])
            Pdel_w.append(df['Pdel_w'][i])
            P_w.append(df['P_w'][i])
            P_s.append(df['P_s'][i])
            Qdel_s.append(df['Qdel_s'][i])
            Qdel_w.append(df['Qdel_w'][i])
            Q_s.append(df['Q_s'][i])
            Q_w.append(df['Q_w'][i])
        f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
        mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw = \
            calculate_yvals(x_column, X_axis, Pdel_s, Pdel_w, P_s, P_w, Q_s, Q_w, Qdel_s, Qdel_w,
                                f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw,
                                mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw)
    return f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
           mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw


# Plot a single panel in the bootstrapping figures
def plot_bootstrapping(ax, X_axis, y_axis, lim=True):
    ax.grid()
    p_2_5 = []
    p_5 = []
    p_10 = []
    p_90 = []
    p_95 = []
    p_97_5 = []
    for i in y_axis:
        p_2_5.append(float(np.percentile(i, [2.5])))
        p_5.append(float(np.percentile(i, [5])))
        p_10.append(float(np.percentile(i, [10])))
        p_90.append(float(np.percentile(i, [90])))
        p_95.append(float(np.percentile(i, [95])))
        p_97_5.append(float(np.percentile(i, [97.5])))
    ax.fill_between(X_axis, p_2_5, p_5, color='yellow', label='2.5th and 97.5th percentiles')
    ax.fill_between(X_axis, p_95, p_97_5, color='yellow')
    ax.fill_between(X_axis, p_5, p_10, color='green', label='5th and 95th percentiles')
    ax.fill_between(X_axis, p_10, p_95, color='green')
    ax.fill_between(X_axis, p_10, p_90, color='blue', label='10th and 90th percentiles')
    #mean_y = [0] * len(X_axis)
    median_y = [0] * len(X_axis)
    for i in range(len(X_axis)):
        #mean_y[i] = sum(y_axis[i])/len(y_axis[i])
        median_y[i] = statistics.median(y_axis[i])
    #plt.plot(X_axis, mean_y, label='Mean', color="orange")
    ax.plot(X_axis, median_y, label='Median', color="black")
    if lim == True:
        ax.set_ylim([-0.5, 1.5])


# Similar to calculate_yvals function above, but using the partitioning equations for the fraction of ET from
# summer precipitation and the fraction of seasonal precipitation to ET (EMS)
def calculate_yvals_et(x_column, x_axis, Pdel_s, Pdel_w, P_s, P_w, Q, Qdel, ET, f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET,
                       mm_Ps_to_ET, mm_Pw_to_ET):
    if x_column == 'P_s':
        X = P_s
    elif x_column == 'P_w':
        X = P_w
    pdels, pdels_se = ols_slope_int(X, Pdel_s)
    pdelw, pdelw_se = ols_slope_int(X, Pdel_w)
    et, et_se = ols_slope_int(X, ET)
    q, q_se = ols_slope_int(X, Q)
    qdel, qdel_se = ols_slope_int(X, Qdel)

    def regression(mb):
        result = mb[0] * x_axis[i] + mb[1]
        return result

    def calc_f_et_from_ps(ps, q, qdel, pdelw, pdels, pw):
        return (ps - q * (qdel - pdelw) / (pdels - pdelw)) / (ps + pw - q)

    def calc_p_to_et(f_et_from_p, et, p):
        return f_et_from_p * et / p

    if x_column == 'P_s':
        pw, pw_se = ols_slope_int(X, P_w)
        for i in range(len(x_axis)):
            f_ET_Ps = calc_f_et_from_ps(x_axis[i], regression(q), regression(qdel), regression(pdelw),
                                        regression(pdels), regression(pw))
            f_Ps_ET = calc_p_to_et(f_ET_Ps, regression(et), x_axis[i])
            f_Pw_ET = calc_p_to_et((1 - f_ET_Ps), regression(et), regression(pw))
            f_ET_from_Ps[i].append(f_ET_Ps)
            f_Ps_to_ET[i].append(f_Ps_ET)
            f_Pw_to_ET[i].append(f_Pw_ET)
            mm_Ps_to_ET[i].append(f_Ps_ET * x_axis[i])
            mm_Pw_to_ET[i].append(f_Pw_ET * regression(pw))
    if x_column == 'P_w':
        ps, ps_se = ols_slope_int(P_w, P_s)
        for i in range(len(x_axis)):
            f_ET_Ps = calc_f_et_from_ps(regression(ps), regression(q), regression(qdel), regression(pdelw),
                                        regression(pdels), x_axis[i])
            f_Ps_ET = calc_p_to_et(f_ET_Ps, regression(et), regression(ps))
            f_Pw_ET = calc_p_to_et((1 - f_ET_Ps), regression(et), x_axis[i])
            f_ET_from_Ps[i].append(f_ET_Ps)
            f_Ps_to_ET[i].append(f_Ps_ET)
            f_Pw_to_ET[i].append(f_Pw_ET)
            mm_Ps_to_ET[i].append(f_Ps_ET * regression(ps))
            mm_Pw_to_ET[i].append(f_Pw_ET * x_axis[i])
    return f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET


def bootstrapping_numerical_et(df, x_column, X_axis):
    # y values used to calculate percentiles and medians for each segment of the numerical solution plot
    f_ET_from_Ps = [[] for _ in X_axis]
    f_Ps_to_ET = [[] for _ in X_axis]
    f_Pw_to_ET = [[] for _ in X_axis]
    mm_Ps_to_ET = [[] for _ in X_axis]
    mm_Pw_to_ET = [[] for _ in X_axis]
    for n in range(3000):     # change to 3000
        Q = []
        Qdel = []
        Pdel_s = []
        Pdel_w = []
        ET = []
        P_w = []
        P_s = []
        length = len(df[x_column])
        random.seed(n)
        sample_indices = choices(range(len(df[x_column])), k=length)
        for i in sample_indices:
            Q.append(df['Q'][i])
            Qdel.append(df['Qdel'][i])
            Pdel_s.append(df['Pdel_s'][i])
            Pdel_w.append(df['Pdel_w'][i])
            ET.append(df['ET'][i])
            P_w.append(df['P_w'][i])
            P_s.append(df['P_s'][i])
        f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET = \
            calculate_yvals_et(x_column, X_axis, Pdel_s, Pdel_w, P_s, P_w, Q, Qdel, ET,
                                f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET)
    return f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET


def plot_panels_q(iso_data, fluxes, pwt, qwt, X_axis, title):
    columns = ['Year', 'Ptot', 'P_s', 'P_s_se', 'P_w', 'P_w_se', 'Pdel_s', 'Pdel_w', 'Q', 'Qdel', 'ET', 'Qdel_s',
               'Qdel_w', 'Q_s', 'Q_w', 'ET_se', 'Q_s_se', 'Q_w_se']
    endsplit_results = []
    for i in range(len(iso_data)):
        row = endsplit(iso_data[i]['Pdel'], iso_data[i]['Qdel'], pwt[i], qwt[i],
                       iso_data[i]['Pdelcat'], iso_data[i]['Qdelcat'], fluxes[i]['P'], fluxes[i]['Q'],
                       fluxes[i]['Pcat'], fluxes[i]['Qcat'])[0]
        row[0] = iso_data[i]['year']
        endsplit_results.append(row)
    df = pd.DataFrame(data=endsplit_results, columns=columns)

    f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
    mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw = bootstrapping_numerical(df, 'P_s', X_axis)

    results = f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
    mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw
    result_cols = ['f_Qs_from_Ps', 'f_Qw_from_Ps', 'f_Ps_to_Qs', 'f_Ps_to_Qw', 'f_Pw_to_Qs', 'f_Pw_to_Qw', \
    'mm_Ps_to_Qs', 'mm_Ps_to_Qw', 'mm_Pw_to_Qs', 'mm_Pw_to_Qw']
    summer_results = {result_cols[i]: results[i] for i in range(len(results))}

    fig, axs = plt.subplots(5, 3, figsize=(9, 14))
    y_ticks = [-0.5, 0, 0.5, 1.0, 1.5]
    axs[0, 0].set_yticks(y_ticks)
    axs[1, 0].set_yticks(y_ticks)
    axs[2, 0].set_yticks(y_ticks)
    axs[0, 1].set_yticks(y_ticks)
    axs[1, 1].set_yticks(y_ticks)
    axs[2, 1].set_yticks(y_ticks)
    axs[0, 2].set_yticks(y_ticks)
    axs[1, 2].set_yticks(y_ticks)
    axs[2, 2].set_yticks(y_ticks)
    axs[3, 0].set_ylim(-100, 850)
    axs[3, 1].set_ylim(-100, 850)
    axs[3, 2].set_ylim(-100, 850)
    axs[4, 0].set_ylim(-100, 850)
    axs[4, 1].set_ylim(-100, 850)
    axs[4, 2].set_ylim(-100, 850)

    axs[0, 1].set_ylabel(('\n'.join(wrap('Fraction of Q$_{S}$ from P$_{S}$ (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[0, 1], X_axis, f_Qs_from_Ps)
    axs[1, 1].set_ylabel(('\n'.join(wrap('Fraction of P$_{S}$ to Q$_{S}$ (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[1, 1], X_axis, f_Ps_to_Qs)
    axs[2, 1].set_ylabel(('\n'.join(wrap('Fraction of P$_{W}$ to Q$_{S}$ (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[2, 1], X_axis, f_Pw_to_Qs)
    axs[3, 1].set_ylabel(('\n'.join(wrap('Amount of P$_{S}$ to Q$_{S}$ (mm)', 30))), fontsize=16)
    plot_bootstrapping(axs[3, 1], X_axis, mm_Ps_to_Qs, lim=False)
    axs[4, 1].set_ylabel(('\n'.join(wrap('Amount of P$_{W}$ to Q$_{S}$ (mm)', 30))), fontsize=16)
    plot_bootstrapping(axs[4, 1], X_axis, mm_Pw_to_Qs, lim=False)
    axs[4, 1].set_xlabel('P$_{S}$ (mm)', fontsize=16)

    f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
    mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw = bootstrapping_numerical(df, 'P_w', X_axis)

    results = f_Qs_from_Ps, f_Qw_from_Ps, f_Ps_to_Qs, f_Ps_to_Qw, f_Pw_to_Qs, f_Pw_to_Qw, \
              mm_Ps_to_Qs, mm_Ps_to_Qw, mm_Pw_to_Qs, mm_Pw_to_Qw
    winter_results = {result_cols[i]: results[i] for i in range(len(results))}

    axs[0, 0].set_ylabel(('\n'.join(wrap('Fraction of Q$_{W}$ from P$_{S}$ (unitless)', 30))), fontsize=16)
    axs[1, 0].set_ylabel(('\n'.join(wrap('Fraction of P$_{S}$ to Q$_{W}$ (unitless)', 30))), fontsize=16)
    axs[2, 0].set_ylabel(('\n'.join(wrap('Fraction of P$_{W}$ to Q$_{W}$ (unitless)', 30))), fontsize=16)
    axs[3, 0].set_ylabel(('\n'.join(wrap('Amount of P$_{S}$ to Q$_{W}$ (mm)', 30))), fontsize=16)
    axs[4, 0].set_ylabel(('\n'.join(wrap('Amount of P$_{W}$ to Q$_{W}$ (mm)', 30))), fontsize=16)
    axs[4, 0].set_xlabel('P$_{W}$ (mm)', fontsize=16)

    plot_bootstrapping(axs[0, 0], X_axis, f_Qw_from_Ps)
    plot_bootstrapping(axs[1, 0], X_axis, f_Ps_to_Qw)
    plot_bootstrapping(axs[2, 0], X_axis, f_Pw_to_Qw)
    plot_bootstrapping(axs[3, 0], X_axis, mm_Ps_to_Qw, lim=False)
    plot_bootstrapping(axs[4, 0], X_axis, mm_Pw_to_Qw, lim=False)

    plot_bootstrapping(axs[0, 2], X_axis, f_Qs_from_Ps)
    plot_bootstrapping(axs[1, 2], X_axis, f_Ps_to_Qs)
    plot_bootstrapping(axs[2, 2], X_axis, f_Pw_to_Qs)
    plot_bootstrapping(axs[3, 2], X_axis, mm_Ps_to_Qs, lim=False)
    plot_bootstrapping(axs[4, 2], X_axis, mm_Pw_to_Qs, lim=False)
    axs[4, 2].set_xlabel('P$_{W}$ (mm)', fontsize=16)

    axs[0, 0].text(450, 1.55, "A", fontsize=16)
    axs[0, 1].text(450, 1.55, "B", fontsize=16)
    axs[0, 2].text(450, 1.55, "C", fontsize=16)
    axs[1, 0].text(450, 1.55, "D", fontsize=16)
    axs[1, 1].text(450, 1.55, "E", fontsize=16)
    axs[1, 2].text(450, 1.55, "F", fontsize=16)
    axs[2, 0].text(450, 1.55, "G", fontsize=16)
    axs[2, 1].text(450, 1.55, "H", fontsize=16)
    axs[2, 2].text(450, 1.55, "I", fontsize=16)
    axs[3, 0].text(450, 890, "J", fontsize=16)
    axs[3, 1].text(450, 875, "K", fontsize=16)
    axs[3, 2].text(450, 875, "L", fontsize=16)
    axs[4, 0].text(450, 875, "M", fontsize=16)
    axs[4, 1].text(450, 875, "N", fontsize=16)
    axs[4, 2].text(450, 875, "O", fontsize=16)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    axs[4, 2].legend(ncol=2, bbox_to_anchor=(0.9, -0.25), fontsize=16)
    #fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\Qpartitioning' + title + '.svg', dpi=500)
    plt.show()

    return df, summer_results, winter_results

def plot_panels_et(years, iso_data, fluxes, pwt, qwt, et_df, X_axis, title):
    columns = ['Year', 'Ptot', 'P_s', 'P_s_se', 'P_w', 'P_w_se', 'Pdel_s', 'Pdel_w', 'Q', 'Qdel', 'ET', 'Qdel_s',
               'Qdel_w', 'Q_s', 'Q_w', 'ET_se', 'Q_s_se', 'Q_w_se']
    annual_precip_upp = [0] * len(years)
    annual_runoff_upp = [0] * len(years)
    for i in range(len(years)):
        for d in range(len(fluxes[i]['P'])):
            annual_precip_upp[i] += fluxes[i]['P'][d]
        for d in range(len(fluxes[i]['Q'])):
            annual_runoff_upp[i] += fluxes[i]['Q'][d]

    lysimeter_et = []
    for i in range(len(years)):
        if years[i] in et_df['Year'].tolist():
            lysimeter_et.append(et_df['annual_ET'][et_df['Year'].tolist().index(years[i])])

    lys_scaled_et = []
    lys_scaled_q = []
    for i in range(len(years)):
        lys_scaled_et.append(lysimeter_et[i] / sum(lysimeter_et) * (sum(annual_precip_upp) - sum(annual_runoff_upp)))
        lys_scaled_q.append(annual_precip_upp[i] - lys_scaled_et[i])

    endsplit_upp = []
    for i in range(len(years)):
        row = endsplit(iso_data[i]['Pdel'], iso_data[i]['Qdel'], pwt[i], qwt[i], iso_data[i]['Pdelcat'],
                       iso_data[i]['Qdelcat'], fluxes[i]['P'], fluxes[i]['Q'], fluxes[i]['Pcat'], fluxes[i]['Qcat'],
                       lys_scaled_et[i], lys_scaled_q[i])[0]
        row[0] = years[i]
        endsplit_upp.append(row)
    df = pd.DataFrame(data=endsplit_upp, columns=columns)

    f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET = bootstrapping_numerical_et(df, 'P_s', X_axis)

    result_cols = ['f_ET_from_Ps', 'f_Ps_to_ET', 'f_Pw_to_ET', 'mm_Ps_to_ET', 'mm_Pw_to_ET']
    results = f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET
    summer_results = {result_cols[i]: results[i] for i in range(len(results))}

    fig, axs = plt.subplots(5, 2, figsize=(6, 14))

    y_ticks = [-0.5, 0, 0.5, 1.0, 1.5]
    axs[0, 0].set_yticks(y_ticks)
    axs[1, 0].set_yticks(y_ticks)
    axs[2, 0].set_yticks(y_ticks)
    axs[0, 1].set_yticks(y_ticks)
    axs[1, 1].set_yticks(y_ticks)
    axs[2, 1].set_yticks(y_ticks)
    axs[3, 0].set_ylim(-300, 800)
    axs[3, 1].set_ylim(-300, 800)
    axs[4, 0].set_ylim(-300, 800)
    axs[4, 1].set_ylim(-300, 800)

    axs[0, 0].set_ylabel(('\n'.join(wrap('Fraction of ET from P$_{S}$ (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[0, 0], X_axis, f_ET_from_Ps)
    axs[1, 0].set_ylabel(('\n'.join(wrap('Fraction of P$_{S}$ to ET (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[1, 0], X_axis, f_Ps_to_ET)
    axs[2, 0].set_ylabel(('\n'.join(wrap('Fraction of P$_{W}$ to ET (unitless)', 30))), fontsize=16)
    plot_bootstrapping(axs[2, 0], X_axis, f_Pw_to_ET)
    axs[3, 0].set_ylabel(('\n'.join(wrap('Amount of P$_{S}$ to ET (mm)', 25))), fontsize=16)
    plot_bootstrapping(axs[3, 0], X_axis, mm_Ps_to_ET, lim=False)
    axs[4, 0].set_ylabel(('\n'.join(wrap('Amount of P$_{W}$ to ET (mm)', 25))), fontsize=16)
    plot_bootstrapping(axs[4, 0], X_axis, mm_Pw_to_ET, lim=False)
    axs[4, 0].set_xlabel('P$_{S}$ (mm)', fontsize=16)

    f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET = bootstrapping_numerical_et(df, 'P_w', X_axis)

    results = f_ET_from_Ps, f_Ps_to_ET, f_Pw_to_ET, mm_Ps_to_ET, mm_Pw_to_ET
    winter_results = {result_cols[i]: results[i] for i in range(len(results))}

    plot_bootstrapping(axs[0, 1], X_axis, f_ET_from_Ps)
    plot_bootstrapping(axs[1, 1], X_axis, f_Ps_to_ET)
    plot_bootstrapping(axs[2, 1], X_axis, f_Pw_to_ET)
    plot_bootstrapping(axs[3, 1], X_axis, mm_Ps_to_ET, lim=False)
    plot_bootstrapping(axs[4, 1], X_axis, mm_Pw_to_ET, lim=False)
    axs[4, 1].set_xlabel('P$_{W}$ (mm)', fontsize=16)

    axs[0, 0].text(450, 1.55, "A", fontsize=16)
    axs[0, 1].text(450, 1.55, "B", fontsize=16)
    axs[1, 0].text(450, 1.55, "C", fontsize=16)
    axs[1, 1].text(450, 1.55, "D", fontsize=16)
    axs[2, 0].text(450, 1.55, "E", fontsize=16)
    axs[2, 1].text(450, 1.55, "F", fontsize=16)
    axs[3, 0].text(450, 820, "G", fontsize=16)
    axs[3, 1].text(450, 820, "H", fontsize=16)
    axs[4, 0].text(450, 820, "I", fontsize=16)
    axs[4, 1].text(450, 835, "J", fontsize=16)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    axs[4, 1].legend(ncol=1, bbox_to_anchor=(0.7, -0.25), fontsize=16)
    #fig.savefig(r'C:\Users\User\Documents\UNR\Swiss Project\Coding\figures\ETpartitioning' + title + '.svg', dpi=500)

    plt.show()

    return df, summer_results, winter_results

