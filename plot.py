from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Plot the amount of summer precipitation by the fraction of evapotranspiration from summer precipitation
# Also print linear regression best fit equation and R squared value
def PlotFraction(summer_P, summer_P_se, f_ET_from_summer, f_ET_se, method, catchment):
    plt.scatter(summer_P, f_ET_from_summer, marker='o', linestyle='None')
    plt.errorbar(summer_P, f_ET_from_summer, xerr=summer_P_se, yerr=f_ET_se, marker='.', linestyle='None')
    plt.title(str(method) + ' Precipitation Values, ' + str(catchment) + " Rietholzbach")
    plt.xlabel('Summer Precipitation (mm)')
    plt.ylabel('Fraction of Evapotranspiration from Summer Precipitation')
    plt.grid()

    reg = LinearRegression().fit(np.array(summer_P).reshape(-1, 1), f_ET_from_summer)
    print('')
    print("Using " + str(method) + " precipitation isotope values, " + str(catchment) + " Rietholzbach catchment")
    print("R squared for summer precipitation v. fraction ET from summer precipitation:",
          reg.score(np.array(summer_P).reshape(-1, 1), f_ET_from_summer))
    print("Equation: y=", reg.coef_, "*x +", reg.intercept_)

    plt.plot(summer_P, reg.coef_ * summer_P + reg.intercept_)
    plt.show()

# Plot the amount of summer precipitation by the amount of evapotranspiration
# Also print linear regression best fit equation and R squared value
def PlotAmount(summer_P, summer_P_se, ET, ET_se, method, catchment):
    plt.scatter(summer_P, ET, marker='o', linestyle='None')
    plt.errorbar(summer_P, ET, xerr=summer_P_se, yerr=ET_se, marker='.', linestyle='None')
    plt.title(str(method) + ' Precipitation Values, ' + str(catchment) + " Rietholzbach")
    plt.xlabel('Summer Precipitation (mm)')
    plt.ylabel('Evapotranspiration (mm)')
    plt.grid()

    reg_et = LinearRegression().fit(np.array(summer_P).reshape(-1, 1), ET)
    print(' ')
    print("Using " + str(method) + " precipitation isotope values, " + str(catchment) + " Rietholzbach catchment")
    print("R squared for summer precipitation v. ET amount:", reg_et.score(np.array(summer_P).reshape(-1, 1), ET))
    print("Equation: y=", reg_et.coef_, "*x +", reg_et.intercept_)

    plt.plot(summer_P, reg_et.coef_ * summer_P + reg_et.intercept_)
    plt.show()

