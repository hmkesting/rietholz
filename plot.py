from sklearn.linear_model import LinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as scipy
import scipy.odr as odr
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Plot the amount of summer precipitation by the fraction of evapotranspiration from summer precipitation
# Also print linear regression best fit equation and R squared value
def PlotFraction(summer_P, summer_P_se, f_ET_from_summer, f_ET_se, method, catchment):
    summer_P, f_ET_from_summer, f_ET_se = zip(*sorted(zip(summer_P, f_ET_from_summer, f_ET_se)))

    # If the p-value of the slope of the weighted regression is lower than 0.1, we will plot it
    # If not, we will only plot the Ordinary Least Squares regression

    w = []
    plot_errors = []
    for i in f_ET_se:
        w.append(1 / i ** 2)
        if i > 10:
            plot_errors.append(0)
        else:
            plot_errors.append(i)

    X = sm.add_constant(summer_P)
    res_wls = sm.WLS(f_ET_from_summer, X, weights=w).fit()
    slope_pval = res_wls.pvalues[1]

    if slope_pval < 0.1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle(str(method) + ", " + str(catchment) + " Rietholzbach Linear Regressions")
        plt.sca(ax1)
        plt.title('Ordinary Least Squares')
        ax1.set(xlabel='Summer Precipitation (mm)', ylabel='Fraction of ET from Summer')
        ax1.scatter(summer_P, f_ET_from_summer, marker='o', linestyle='None')
        ax1.errorbar(summer_P, f_ET_from_summer, xerr=summer_P_se, yerr=plot_errors, marker='.', linestyle='None')

        reg = sm.OLS(f_ET_from_summer, X).fit()
        ax1.plot(summer_P, reg.fittedvalues)                  #params[1] * np.array(summer_P) + reg.params[0])
        st, data, ss2 = summary_table(reg, alpha=0.025)
        mean_ci_low, mean_ci_upp = data[:, 4:6].T
        ax1.plot(summer_P, mean_ci_low)
        ax1.plot(summer_P, mean_ci_upp)

        plt.sca(ax2)
        plt.title('Inverse Square Error Weighted')
        ax2.set(xlabel='Summer Precipitation (mm)')
        # ax2.set_ylim([0, 1])

        point_size = []
        for i in w:
            point_size.append(math.sqrt(i) * 10)

        ax2.scatter(summer_P, f_ET_from_summer, point_size, marker='o', linestyle='None')
        ax2.plot(summer_P, res_wls.fittedvalues)
        wtd_sum_x = []
        for i in range(len(w)):
            wtd_sum_x.append(w[i]*summer_P[i])
        wtd_mean_x = sum(wtd_sum_x)/sum(w)
        int_se, slope_se = res_wls.bse
        df = res_wls.df_resid
        t_crit = abs(scipy.stats.t.ppf(q=0.025, df=df))
        x_val_ci = range(450, 1050, 25)
        ci_upp = [0] * len(x_val_ci)
        ci_low = [0] * len(x_val_ci)
        for i in range(len(x_val_ci)):
            ci_upp[i] = res_wls.params[1] * x_val_ci[i] + res_wls.params[0] + t_crit * math.sqrt((abs(x_val_ci[i] - wtd_mean_x) * slope_se) ** 2 + int_se ** 2)
            ci_low[i] = res_wls.params[1] * x_val_ci[i] + res_wls.params[0] - t_crit * math.sqrt((abs(x_val_ci[i] - wtd_mean_x) * slope_se) ** 2 + int_se ** 2)
        ax2.plot(x_val_ci, ci_low, color='blue')
        ax2.plot(x_val_ci, ci_upp, color='blue')
    else:
        plt.title(str(method) + ", " + str(catchment) + ' Ordinary Least Squares Linear Regression')
        plt.xlabel('Summer Precipitation (mm)')
        plt.ylabel('Fraction of ET from Summer')
        plt.scatter(summer_P, f_ET_from_summer, marker='o', linestyle='None')
        plt.errorbar(summer_P, f_ET_from_summer, xerr=summer_P_se, yerr=plot_errors, marker='.', linestyle='None')

        reg = sm.OLS(f_ET_from_summer, X).fit()
        plt.plot(summer_P, reg.fittedvalues)  # params[1] * np.array(summer_P) + reg.params[0])
        st, data, ss2 = summary_table(reg, alpha=0.025)
        mean_ci_low, mean_ci_upp = data[:, 4:6].T
        plt.plot(summer_P, mean_ci_low)
        plt.plot(summer_P, mean_ci_upp)

        summer_P, res_wls.fittedvalues, x_val_ci, ci_low, ci_upp = [[], [], [], [], []]
    plt.show()
    #plt.savefig(str(method) + '_' + str(catchment) + '_f_ET_s')
    return summer_P, res_wls.fittedvalues, x_val_ci, ci_low, ci_upp

# Plot the amount of summer precipitation by the amount of evapotranspiration
# Also print linear regression best fit equation and R squared value
def PlotAmount(summer_P, summer_P_se, ET, ET_se, method, catchment):
    plt.scatter(summer_P, ET, marker='o', linestyle='None')
    plt.errorbar(summer_P, ET, xerr=summer_P_se, yerr=ET_se, marker='.', linestyle='None')
    plt.title(str(method) + ', ' + str(catchment) + " Rietholzbach")
    plt.xlabel('Summer Precipitation (mm)')
    plt.ylabel('Evapotranspiration (mm)')

    reg_et = LinearRegression().fit(np.array(summer_P).reshape(-1, 1), ET)
    #print(' ')
    #print("Using " + str(method) + " precipitation isotope values, " + str(catchment) + " Rietholzbach catchment")
    #print("R squared for summer precipitation v. ET amount:", reg_et.score(np.array(summer_P).reshape(-1, 1), ET))
    #print("Equation: y=", reg_et.coef_, "*x +", reg_et.intercept_)

    plt.plot(summer_P, reg_et.coef_ * summer_P + reg_et.intercept_)
    plt.show()
    #plt.savefig(str(method) + '_' + str(catchment) + '_ET_amt')

def line(x, a):
    y=a[0]*x+a[1]
    return y

def odr_line(p, x):
    y=[0]*len(x)
    for i in range(len(x)):
        y[i] = p[0]*x[i]+p[1]
    return np.array(y)

def perform_odr(x, y, xerr, yerr):
    linear = odr.Model(odr_line)
    mydata = odr.Data(x, y, wd=xerr, we=yerr)
    myodr = odr.ODR(mydata, linear, beta0=np.array([0, 0]))
    output = myodr.run()
    return output

def PlotODR(x, y, method):
    x_input = x.loc[:, 'f_ET']
    y_input = y.loc[:, 'f_ET']
    xerr_input = x.loc[:, 'f_ET_se']
    yerr_input = y.loc[:, 'f_ET_se']

    subset_x = []
    subset_xerr = []
    for i in range(len(x)):
        if x.loc[i, 'Year'] in list(y.loc[:, 'Year']):
            subset_x.append(x_input[i])
            subset_xerr.append(xerr_input[i])

    plot_xerr = []
    plot_yerr = []
    for i in range(len(yerr_input)):
        plot_xerr.append(subset_xerr[i])
        plot_yerr.append(yerr_input[i])
        if subset_xerr[i] > 10:
            plot_xerr[i] = 0
        if yerr_input[i] > 10:
            plot_yerr[i] = 0

    plt.scatter(subset_x, y_input, marker='o', linestyle='None')
    plt.errorbar(subset_x, y_input, xerr=plot_xerr, yerr=plot_yerr, marker='.', linestyle='None')
    plt.title("Fraction of ET from Summer Precipitation, " + method + " Rietholzbach")
    plt.xlabel('Fraction of ET from Summer Using Lysimeter')
    plt.ylabel('Fraction of ET from Summer Using Streamflow')

    x_err = []
    y_err = []

    for i in range(len(subset_x)):
        x_err.append(1./subset_xerr[i])
        y_err.append(1./yerr_input.loc[i])

    xerr=np.array(x_err)
    yerr=np.array(y_err)

    x=np.array(subset_x)
    y=np.array(y_input)

    plt.plot([min(subset_x), max(subset_x)], [min(subset_x), max(subset_x)], color='k', linestyle='dashed')

    reg = LinearRegression().fit(np.array(subset_x).reshape(-1, 1), y_input)
    plt.plot(subset_x, reg.coef_ * subset_x + reg.intercept_, label='Least Squares')

    regression = perform_odr(x, y, xerr, yerr)
    plt.plot(x, line(x, regression.beta), label='ODR')

    #plt.xlim(axis_limits[0], axis_limits[1])
    #plt.ylim(axis_limits[2], axis_limits[3])

    plt.legend()
    plt.show()
    #plt.savefig(str(method) + '_f_ET_s_compare')
    return x, regression.beta

def LysCorrectedEndsplit(watershed, lysimeter, method, catchment, undercatch):
    P_s = watershed['P_s']
    years = watershed['Year']
    Qdel = watershed['Qdel']
    Pdel_s = watershed['Pdel_s']
    Pdel_w = watershed['Pdel_w']
    AllP_del = watershed['AllP_del']
    Ptot = watershed['Ptot']
    for i in range(len(years)):
        if years[i] in undercatch['Year']:
            Ptot[i] += undercatch['Undercatch (mm)'][undercatch['Year'].index(years[i])]
    Q_post = [0]*len(years)
    ETdel = [0]*len(years)
    f_ET  = [0]*len(years)
    for i in range(len(years)):
        Q_post[i] = Ptot[i] - lysimeter
        ETdel[i] = AllP_del[i] * Ptot[i] - Qdel[i] * Q_post[i] / lysimeter
        f_ET[i] = (ETdel[i]-Pdel_w[i])/(Pdel_s[i]-Pdel_w[i])

    plt.scatter(P_s, f_ET, marker='o', linestyle='None')
    #plt.errorbar(P_s, f_ET, xerr=P_s_err, yerr=f_ET_err, marker='.', linestyle='None')
    plt.title("Lysimeter-Corrected Streamflow, " + method + ' ' + catchment + " Rietholzbach")
    plt.xlabel('Summer Precipitation (mm)')
    plt.ylabel('Fraction of ET from Summer')

    reg = LinearRegression().fit(np.array(P_s).reshape(-1, 1), f_ET)

    plt.plot(P_s, reg.coef_ * P_s + reg.intercept_)

    plt.show()

    #plt.savefig(str(method) + '_' + str(catchment) + '_' + 'f_ET_s_lys-corrected')
    return f_ET