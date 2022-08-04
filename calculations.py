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
def data_checks(pcat, pdelcat, pdel, pwt, p, qcat, qdelcat, qdel, qwt, q):
    if set(pcat) != set(pdelcat):
        raise Exception('fatal error: Pcat and Pdelcat use different labels')
    if set(qcat) != set(qdelcat):
        raise Exception('fatal error: Qcat and Qdelcat use different labels')
    if len(set(pcat)) != 2:
        raise Exception('fatal error: need exactly two precipitation categories')

    n = len(pdel)
    if len(pwt) != n or len(pdelcat) != n:
        raise Exception("fatal error: Pdel, Pwt, and Pdelcat must have the same length")
    n = len(qdel)
    if len(qwt) != n or len(qdelcat) != n:
        raise Exception("fatal error: Qdel, Qwt, and Qdelcat must have the same length")
    if len(p) != len(pcat):
        raise Exception("fatal error: P and Pcat must have the same length")
    if len(q) != len(qcat):
        raise Exception("fatal error: Q and Qcat must have the same length")

# Calculate isotope weighted means, total fluxes, and associated errors
def calc_isotopes_and_fluxes(isotope, isotope_category, isotope_weight, flux, flux_category):
    isotope_means = [0, 0]
    isotope_error = [0, 0]
    flux_totals = [0, 0]
    flux_error = [0, 0]

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

    all_flux = flux_totals[0] + flux_totals[1]
    all_flux_se = math.sqrt(flux_error[0]**2 + flux_error[1]**2)
    all_flux_del = (isotope_means[0] * flux_totals[0] + isotope_means[1] * flux_totals[1])/all_flux
    all_flux_del_se = math.sqrt(((isotope_error[0] * flux_totals[0]/ all_flux)**2 + (isotope_error[1] * flux_totals[1]/all_flux)**2
                            + (isotope_means[0] * flux_error[0] * (all_flux - flux_totals[0])/all_flux**2)**2
                            + (isotope_means[1] * flux_error[1] * (all_flux - flux_totals[1])/all_flux**2)**2))
    return isotope_means, isotope_error, flux_totals, flux_error, all_flux, all_flux_se, all_flux_del, all_flux_del_se

# Calculate the isotope value and amount of evapotranspiration with associated errors
def calc_et_values(allp, allq, allp_se, allq_se, allp_del, allq_del, pdel_bar, ptot, pdel_se, allq_del_se, ptot_se):
    et = allp - allq
    et_se = math.sqrt(allp_se ** 2 + allq_se ** 2)
    et_del = (allp_del * allp - allq_del * allq) / et

    d_d_pdel1 = ptot[0] / et
    d_d_pdel2 = ptot[1] / et
    d_d_allq_del = -allq / et
    d_d_ptot1 = (pdel_bar[0] - et_del) / et
    d_d_ptot2 = (pdel_bar[1] - et_del) / et
    d_d_allq = (et_del - allq_del) / et

    et_del_se = math.sqrt((pdel_se[0] * d_d_pdel1) ** 2
                            + (pdel_se[1] * d_d_pdel2) ** 2
                            + (allq_del_se * d_d_allq_del) ** 2
                            + (ptot_se[0] * d_d_ptot1) ** 2
                            + (ptot_se[1] * d_d_ptot2) ** 2
                            + (allq_se * d_d_allq) ** 2)

    if math.isnan(et) or math.isnan(et_se) or math.isnan(et_del) or math.isnan(et_del_se):
        raise Exception('NaN outputs')
    return et, et_se, et_del, et_del_se

# Create table with results of end-member mixing
def end_member_mixing(pdel_bar, qdel_bar, pdel_se, qdel_se, ptot, qtot, ptot_se, qtot_se, allq, et):
    f = [[0, 0], [0, 0], [0, 0], [0, 0]]
    f_se = [[0, 0], [0, 0], [0, 0], [0, 0]]

    denom = pdel_bar[0] - pdel_bar[1]
    for j in range(3):
        f[j][0] = (qdel_bar[j] - pdel_bar[1]) / denom
        f[j][1] = 1 - f[j][0]
        f_se[j][0] = math.sqrt((qdel_se[j] / denom) ** 2 + (pdel_se[0] * (-f[j][0] / denom)) ** 2 + (
                    pdel_se[1] * (pdel_bar[0] - qdel_bar[j]) / denom ** 2) ** 2)
        f_se[j][1] = f_se[j][0]

    f[3][0] = (ptot[0] - qtot[2] * (qdel_bar[2] - pdel_bar[1]) / denom) / (ptot[0] + ptot[1] - qtot[2])
    f_se[3][0] = math.sqrt((ptot_se[0] * (1 - f[3][0]) / et) ** 2
                           + (ptot_se[1] * f[3][0] / et) ** 2
                           + (qtot_se[2] * (f[3][0] - f[2][0]) / et) ** 2
                           + (qdel_se[2] * allq / (pdel_bar[0] - pdel_bar[1]) / et) ** 2
                           + (pdel_se[0] * allq * f[2][0] / (pdel_bar[0] - pdel_bar[1]) / et) ** 2
                           + (pdel_se[1] * allq * f[2][1] / (pdel_bar[1] - pdel_bar[0]) / et) ** 2)
    f[3][1] = 1 - f[3][0]
    f_se[3][1] = f_se[3][0]
    return f, f_se

# Create table with results of end-member splitting
def end_member_splitting(qtot, ptot, f, f_se, qtot_se, ptot_se):
    eta = [[0, 0], [0, 0], [0, 0], [0, 0]]
    eta_se = [[0, 0], [0, 0], [0, 0], [0, 0]]

    for j in range(3):
        for i in range(2):
            eta[j][i] = f[j][i] * qtot[j] / ptot[i]
            eta_se[j][i] = abs(eta[j][i]) * math.sqrt((f_se[j][i] / f[j][i]) ** 2
                                                      + (qtot_se[j] / qtot[j]) ** 2 + (ptot_se[i] / ptot[i]) ** 2)

    for i in range(2):
        eta[3][i] = 1 - eta[2][i]
        eta_se[3][i] = eta_se[2][i]
    return eta, eta_se

def format_tables(f, f_se, eta, eta_se):
    f = pd.DataFrame(f, columns=('f.summer', 'f.winter'))
    f_se = pd.DataFrame(f_se, columns=('f.summer.se', 'f.winter.se'))
    eta = pd.DataFrame(eta, columns=('eta.summer', 'eta.winter'))
    eta_se = pd.DataFrame(eta_se, columns=('eta.summer.se', 'eta.winter.se'))

    table = pd.concat([f, f_se, eta, eta_se], axis=1)
    table.index = ['summer', 'winter', 'AllQ', 'ET']

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    return table

def endsplit(pdel, qdel, pwt, qwt, pdelcat, qdelcat, p, q, pcat, qcat):

    data_checks(pcat, pdelcat, pdel, pwt, p, qcat, qdelcat, qdel, qwt, q)

    pdel_bar, pdel_se, ptot, ptot_se, allp, allp_se, allp_del, allp_del_se = calc_isotopes_and_fluxes(pdel, pdelcat, pwt, p, pcat)
    qdel_bar, qdel_se, qtot, qtot_se, allq, allq_se, allq_del, allq_del_se = calc_isotopes_and_fluxes(qdel, qdelcat, qwt, q, qcat)

    et, et_se, et_del, et_del_se = calc_et_values(allp, allq, allp_se, allq_se, allp_del,
                                                                allq_del, pdel_bar, ptot, pdel_se, allq_del_se, ptot_se)

    qdel_bar = np.append(qdel_bar, allq_del)
    qdel_se = np.append(qdel_se, allq_del_se)
    qtot = np.append(qtot, allq)
    qtot_se = np.append(qtot_se, allq_se)

    qdel_bar = np.append(qdel_bar, et_del)
    qdel_se = np.append(qdel_se, et_del_se)
    qtot = np.append(qtot, et)
    qtot_se = np.append(qtot_se, et_se)

    f, f_se = end_member_mixing(pdel_bar, qdel_bar, pdel_se, qdel_se, ptot, qtot, ptot_se, qtot_se, allq, et)

    eta, eta_se = end_member_splitting(qtot, ptot, f, f_se, qtot_se, ptot_se)

    table = format_tables(f, f_se, eta, eta_se)

    f_ET_from_summer = table.iloc[3, 0]       # f ET from Ps is [3, 0] # f Ps to ET is [3, 4]
    f_ET_se = table.iloc[3, 2]                # f ET from Ps is [3, 2] # f Ps to ET is [3, 6]
    f_ps_to_ET = table.iloc[3, 4]
    f_ps_se = table.iloc[3, 6]
    ET = qtot[3]
    ET_se = qtot_se[3]
    allq = qtot[2]
    qdel_bar = qdel_bar[2]
    pdel_s = pdel_bar[0]
    pdel_w = pdel_bar[1]

    ratio_se = ptot[0]/ptot[1]*math.sqrt((ptot_se[0]/ptot[0])**2 + (ptot_se[1]/ptot[1])**2)
    return [0, allq, qdel_bar, allp, ptot[0], ptot_se[0], pdel_s, pdel_w, f_ET_from_summer, f_ET_se, ET, ET_se,
            allp_del, f_ps_to_ET, f_ps_se, ptot[1], ptot_se[1], ptot[0]/ptot[1], ratio_se], table









