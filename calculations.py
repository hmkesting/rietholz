import math
import numpy as np
import pandas as pd

# Function to calculate weighted mean and error
def wtd_mean(x, wt=None):
    if wt is None:
        wt = [1] * len(x)

    remove = []
    for item in range(len(x)):
        if math.isnan(x[item]) or math.isnan(wt[item]):
            remove.append(item)
    for item in sorted(remove, reverse=True):
        del x[item]
        del wt[item]

    if len(x) != len(wt):
        raise Exception("error in wtd_mean: x and wt have different lengths")

    for item in range(len(x)):
        if wt[item] < 0:
            raise Exception("error in wtd_mean: negative weights")

    sumwt = sum(wt)

    sq_list = []
    for item in range(len(x)):
        sq_list.append(wt[item]**2)
    sumsq = sum(sq_list)

    n_eff = (sumwt*sumwt)/sumsq

    xbar_list = []
    for item in range(len(x)):
        xbar_list.append(x[item]*wt[item])
    xbar = sum(xbar_list)/sumwt

    varx_list = []
    for item in range(len(x)):
        varx_list.append(wt[item] * ((x[item] - xbar)**2))
    varx = (sum(varx_list)/sumwt) * n_eff/(n_eff - 1.0)
    return xbar, math.sqrt(varx/n_eff)

# Check the category labels and vector lengths match
def DataChecks(Pcat, Pdelcat, Pdel, Pwt, P, Qcat, Qdelcat, Qdel, Qwt, Q):

    if set(Pcat) != set(Pdelcat):
        raise Exception('fatal error: Pcat and Pdelcat use different labels')
    if set(Qcat) != set(Qdelcat):
        raise Exception('fatal error: Qcat and Qdelcat use different labels')
    if len(set(Pcat)) != 2:
        raise Exception('fatal error: need exactly two precipitation categories')

    n = len(Pdel)
    if len(Pwt) != n or len(Pdelcat) != n:
        raise Exception("fatal error: Pdel, Pwt, and Pdelcat must have the same length")
    n = len(Qdel)
    if len(Qwt) != n or len(Qdelcat) != n:
        raise Exception("fatal error: Qdel, Qwt, and Qdelcat must have the same length")
    if len(P) != len(Pcat):
        raise Exception("fatal error: P and Pcat must have the same length")
    if len(Q) != len(Qcat):
        raise Exception("fatal error: Q and Qcat must have the same length")

def LysimeterUndercatch(flux_data, category='both'):
    if category == 'both':
        for y in range(len(flux_data)):
            for d in range(len(flux_data[y]['P'])):
                if flux_data[y]['Tcat'][d] == 'snow':
                    flux_data[y]['P'][d] = (flux_data[y]['P'][d]*1.5)
                if flux_data[y]['Tcat'][d] == 'rain':
                    flux_data[y]['P'][d] = (flux_data[y]['P'][d]*1.15)

    if category == 'snow':
        for y in range(len(flux_data)):
            for d in range(len(flux_data[y]['P'])):
                if flux_data[y]['Tcat'][d] == 'snow':
                    flux_data[y]['P'][d] = (flux_data[y]['P'][d]*1.5)

    if category == 'rain':
        for y in range(len(flux_data)):
            for d in range(len(flux_data[y]['P'])):
                if flux_data[y]['Tcat'][d] == 'rain':
                    flux_data[y]['P'][d] = (flux_data[y]['P'][d]*1.15)

    return flux_data

# Calculate isotope weighted means, total fluxes, and associated errors
def CalcIsotopesAndFluxes(isotope, isotope_category, isotope_weight, flux, flux_category):
    isotope_means = np.zeros(2, dtype=object)
    isotope_error = np.zeros(2, dtype=object)
    flux_totals = np.zeros(2, dtype=object)
    flux_error = np.zeros(2, dtype=object)

    isotope_summer = []
    isotope_winter = []
    weight_summer = []
    weight_winter = []
    flux_summer = []
    flux_winter = []
    for i in range(len(flux_category)):
        if flux_category[i] == 'summer':
            flux_summer.append(flux[i])
        else:
            flux_winter.append(flux[i])
    for i in range(len(isotope_category)):
        if isotope_category[i] == 'summer':
            isotope_summer.append(isotope[i])
            weight_summer.append(isotope_weight[i])
        else:
            isotope_winter.append(isotope[i])
            weight_winter.append(isotope_weight[i])
    isotope_means[0], isotope_error[0] = wtd_mean(isotope_summer, weight_summer)
    isotope_means[1], isotope_error[1] = wtd_mean(isotope_winter, weight_winter)
    flux_totals[0], flux_error[0] = wtd_mean(flux_summer)
    flux_totals[0] = flux_totals[0] * len(flux_summer)
    flux_error[0] = flux_error[0] * len(flux_summer)
    flux_totals[1], flux_error[1] = wtd_mean(flux_winter)
    flux_totals[1] = flux_totals[1] * len(flux_winter)
    flux_error[1] = flux_error[1] * len(flux_winter)

    All_flux = flux_totals[0] + flux_totals[1]
    All_flux_se = math.sqrt(flux_error[0]**2 + flux_error[1]**2)
    All_flux_del = (isotope_means[0] * flux_totals[0] + isotope_means[1] * flux_totals[1])/ All_flux
    All_flux_del_se = math.sqrt(((isotope_error[0] * flux_totals[0]/ All_flux)**2 + (isotope_error[1] * flux_totals[1]/ All_flux)**2
                            + (isotope_means[0] * flux_error[0] * (All_flux - flux_totals[0])/All_flux**2)**2
                             + (isotope_means[1] * flux_error[1] * (All_flux - flux_totals[1])/All_flux**2)**2))
    return isotope_means, isotope_error, flux_totals, flux_error, All_flux, All_flux_se, All_flux_del, All_flux_del_se

# Calculate the isotope value and amount of evapotranspiration with associated errors
def CalcEvapotranspirationValues(AllP, AllQ, AllP_se, AllQ_se, AllP_del, AllQ_del, Pdel_bar, Ptot, Pdel_se, AllQ_del_se, Ptot_se):
    nonQ = AllP - AllQ
    nonQ_se = math.sqrt(AllP_se ** 2 + AllQ_se ** 2)
    nonQ_del = (AllP_del * AllP - AllQ_del * AllQ) / nonQ

    d_d_Pdel1 = Ptot[0] / nonQ
    d_d_Pdel2 = Ptot[1] / nonQ
    d_d_AllQ_del = -AllQ / nonQ
    d_d_Ptot1 = (Pdel_bar[0] - nonQ_del) / nonQ
    d_d_Ptot2 = (Pdel_bar[1] - nonQ_del) / nonQ
    d_d_AllQ = (nonQ_del - AllQ_del) / nonQ

    nonQ_del_se = math.sqrt((Pdel_se[0] * d_d_Pdel1) ** 2
                            + (Pdel_se[1] * d_d_Pdel2) ** 2
                            + (AllQ_del_se * d_d_AllQ_del) ** 2
                            + (Ptot_se[0] * d_d_Ptot1) ** 2
                            + (Ptot_se[1] * d_d_Ptot2) ** 2
                            + (AllQ_se * d_d_AllQ) ** 2)
    return nonQ, nonQ_se, nonQ_del, nonQ_del_se

# Create table with results of end-member mixing
def EndMemberMixing(Pdel_bar, Qdel_bar, Pdel_se, Qdel_se, Ptot, Qtot, Ptot_se, Qtot_se, AllQ, nonQ):
    f = np.zeros((4, 2))
    f_se = np.zeros((4, 2))

    denom = Pdel_bar[0] - Pdel_bar[1]
    for j in range(3):
        f[j, 0] = (Qdel_bar[j] - Pdel_bar[1]) / denom
        f[j, 1] = 1 - f[j, 0]
        f_se[j, 0] = math.sqrt((Qdel_se[j] / denom) ** 2 + (Pdel_se[0] * (-f[j, 0] / denom)) ** 2 + (
                    Pdel_se[1] * (Pdel_bar[0] - Qdel_bar[j]) / denom ** 2) ** 2)
        f_se[j, 1] = f_se[j, 0]

    f[3, 0] = (Ptot[0] - Qtot[2] * (Qdel_bar[2] - Pdel_bar[1]) / denom) / (Ptot[0] + Ptot[1] - Qtot[2])
    f_se[3, 0] = math.sqrt((Ptot_se[0] * (1 - f[3, 0]) / nonQ) ** 2
                           + (Ptot_se[1] * f[3, 0] / nonQ) ** 2
                           + (Qtot_se[2] * (f[3, 0] - f[2, 0]) / nonQ) ** 2
                           + (Qdel_se[2] * AllQ / (Pdel_bar[0] - Pdel_bar[1]) / nonQ) ** 2
                           + (Pdel_se[0] * AllQ * f[2, 0] / (Pdel_bar[0] - Pdel_bar[1]) / nonQ) ** 2
                           + (Pdel_se[1] * AllQ * f[2, 1] / (Pdel_bar[1] - Pdel_bar[0]) / nonQ) ** 2)
    f[3, 1] = 1 - f[3, 0]
    f_se[3, 1] = f_se[3, 0]
    return f, f_se

# Create table with results of end-member splitting
def EndMemberSplitting(Qtot, Ptot, f, f_se, Qtot_se, Ptot_se):
    eta = np.zeros((4, 2))
    eta_se = np.zeros((4, 2))

    for j in range(3):
        for i in range(2):
            eta[j, i] = f[j, i] * Qtot[j] / Ptot[i]
            eta_se[j, i] = abs(eta[j, i]) * math.sqrt((f_se[j, i] / f[j, i]) ** 2
                                                      + (Qtot_se[j] / Qtot[j]) ** 2 + (Ptot_se[i] / Ptot[i]) ** 2)

    for i in range(2):
        eta[3, i] = 1 - eta[2, i]
        eta_se[3, i] = eta_se[2, i]
    return eta, eta_se

def FormatTables(f, f_se, eta, eta_se):
    f = pd.DataFrame(f, columns=('f.summer', 'f.winter'))
    f_se = pd.DataFrame(f_se, columns=('f.summer.se', 'f.winter.se'))
    eta = pd.DataFrame(eta, columns=('eta.summer', 'eta.winter'))
    eta_se = pd.DataFrame(eta_se, columns=('eta.summer.se', 'eta.winter.se'))

    table = pd.concat([f, f_se, eta, eta_se], axis=1)
    table.index = ['summer', 'winter', 'AllQ', 'nonQ']

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    return table

def EndSplit(Pdel, Qdel, Pwt, Qwt, Pdelcat, Qdelcat, P, Q, Pcat, Qcat):

    DataChecks(Pcat, Pdelcat, Pdel, Pwt, P, Qcat, Qdelcat, Qdel, Qwt, Q)

    Pdel_bar, Pdel_se, Ptot, Ptot_se, AllP, AllP_se, AllP_del, AllP_del_se = CalcIsotopesAndFluxes(Pdel, Pdelcat, Pwt, P, Pcat)
    Qdel_bar, Qdel_se, Qtot, Qtot_se, AllQ, AllQ_se, AllQ_del, AllQ_del_se = CalcIsotopesAndFluxes(Qdel, Qdelcat, Qwt, Q, Qcat)

    nonQ, nonQ_se, nonQ_del, nonQ_del_se = CalcEvapotranspirationValues(AllP, AllQ, AllP_se, AllQ_se, AllP_del,
                                                                AllQ_del, Pdel_bar, Ptot, Pdel_se, AllQ_del_se, Ptot_se)

    Qdel_bar = np.append(Qdel_bar, AllQ_del)
    Qdel_se = np.append(Qdel_se, AllQ_del_se)
    Qtot = np.append(Qtot, AllQ)
    Qtot_se = np.append(Qtot_se, AllQ_se)

    Qdel_bar = np.append(Qdel_bar, nonQ_del)
    Qdel_se = np.append(Qdel_se, nonQ_del_se)
    Qtot = np.append(Qtot, nonQ)
    Qtot_se = np.append(Qtot_se, nonQ_se)

    f, f_se = EndMemberMixing(Pdel_bar, Qdel_bar, Pdel_se, Qdel_se, Ptot, Qtot, Ptot_se, Qtot_se, AllQ, nonQ)

    eta, eta_se = EndMemberSplitting(Qtot, Ptot, f, f_se, Qtot_se, Ptot_se)

    table = FormatTables(f, f_se, eta, eta_se)

    f_ET_from_summer = table.iloc[3, 0]
    f_ET_se = table.iloc[3, 2]
    ET = Qtot[3]
    ET_se = Qtot_se[3]
    summer_P = Ptot[0]
    summer_P_se = Ptot_se[0]
    Qtot = Qtot[2]
    Qdel_bar = Qdel_bar[2]
    Pdel_s = Pdel_bar[0]
    Pdel_w = Pdel_bar[1]
    return [0, Qtot, Qdel_bar, AllP, summer_P, summer_P_se, Pdel_s, Pdel_w, f_ET_from_summer, f_ET_se, ET, ET_se, AllP_del]










