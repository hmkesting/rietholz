import pandas as pd
import numpy as np
from preprocessing import convert_datetime, list_years, find_range_sampling, list_unusable_dates, \
    split_fluxes_by_hydro_year, split_isotopes_by_hydro_year
from cleaning import sum_precipitation_and_runoff, remove_nan_samples
from calculations import wtd_mean, data_checks, calc_isotopes_and_fluxes, calc_et_values, end_member_mixing, \
    end_member_splitting, format_tables, endsplit
from delta_figure import to_year_fraction, calc_precip, calc_q
from plot import calculate_wls, undercatch_correction, calculate_fractions, round_half_up, plot_panels, plot_et_amounts, \
    calculate_avg_et, calculate_scaled_et
from main_endsplit import workflow_endsplit, confidence_intervals
import unittest
from datetime import datetime as dt

class TestConvertDatetime(unittest.TestCase):
    def test_convert_datetime1(self):
        self.assertEqual(convert_datetime(['08/23/21', '09/24/21']), [dt(2021, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_convert_datetime2(self):
        self.assertEqual(convert_datetime(['08/23/2021', '09/24/2021']), [dt(2021, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_convert_datetime3(self):
        self.assertEqual(convert_datetime(['08/23/73', '09/24/21']), [dt(1973, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_convert_datetime4(self):
        with self.assertRaises(Exception):
            convert_datetime(['08/23/21', '24/09/21'])
    def test_convert_datetime5(self):
        with self.assertRaises(Exception):
            convert_datetime(['08/23/221', '09/24/021'])
    def test_convert_datetime6(self):
        with self.assertRaises(ValueError):
            convert_datetime(['hello', '09/24/021'])
    def test_convert_datetime7(self):
        with self.assertRaises(Exception):
            convert_datetime(['', '09/24/021'])

class TestListYears(unittest.TestCase):
    def test_list_years1(self):
        self.assertEqual(list_years((dt(2019, 3, 14, 0, 0), dt(2003, 10, 27, 0, 0))), [2019, 2003])
    def test_list_years2(self):
        self.assertEqual(list_years([dt(2003, 3, 14, 0, 0), dt(2023, 10, 27, 0, 0)]), [2003, 2023])
    def test_list_years3(self):
        with self.assertRaises(Exception):
            list_years(['06/26/2006'])

class TestFindRangeSampling(unittest.TestCase):
    def test_find_range_sampling1(self):
        self.assertEqual(find_range_sampling([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], [dt(1992, 4, 14, 0, 0), dt(2008, 10, 27, 0, 0)], [-12.456, -15], [456, 586], [-11, -12.0], [434, 632], side='start'), (4, 1993))
    def test_find_range_sampling2(self):
        self.assertEqual(find_range_sampling([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], [dt(1992, 4, 14, 0, 0), dt(2008, 10, 27, 0, 0)], [np.NaN, 15], [456, 586], [-11, -12.0], [434, 632], side='start'), (10, 2003))
    def test_find_range_sampling3(self):
        self.assertEqual(find_range_sampling([dt(2003, 10, 27, 0, 0), dt(1993, 4, 14, 0, 0)], [dt(2008, 10, 27, 0, 0), dt(1992, 4, 14, 0, 0)], [-12.456, -15], [456, 586], [-11, -12.0], [434, 632], side='end'), (10, 2003))
    def test_find_range_sampling4(self):
        self.assertEqual(find_range_sampling([dt(2003, 10, 27, 0, 0), dt(1993, 4, 14, 0, 0)], [dt(2008, 10, 27, 0, 0), dt(1992, 4, 14, 0, 0)], [-9, 15], [np.NAN, 456], [-11, -12.0], [434, 632], side='end'), (4, 1992))
    def test_find_range_sampling5(self):
        with self.assertRaises(TypeError):
            find_range_sampling([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], ['hello', -15], [-13, -12.0], [93, 3]), (4, 0)

class TestListUnusableDates(unittest.TestCase):
    def test_list_unusable_dates1(self):
        self.assertEqual(list_unusable_dates([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0),  dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], 6, 0, 12, 2, [2012, 2013, 2014]), [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)])
    def test_list_unusable_dates2(self):
        self.assertEqual(list_unusable_dates([dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], 11, 0, 12, 3, [2011, 2012, 2013, 2014]), [dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)])
    def test_list_unusable_dates3(self):
        self.assertEqual(list_unusable_dates([dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0), dt(2015, 1, 19, 0, 0)], 11, 0, 1, 4, [2011, 2012, 2013, 2014, 2015]), [dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0), dt(2015, 1, 19, 0, 0)])

class TestSplitFluxesByHydroYear(unittest.TestCase):
    def test_split_fluxes_by_hydro_year1(self):
        self.assertEqual(split_fluxes_by_hydro_year([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8], [-1, 2, 4, 8, 15, 22, -2, 0]), [{'year': 2012, 'dates':[dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0)], 'P':[0.5, 0.2, 5], 'Pcat': ['winter', 'winter', 'winter'], 'Q': [1, 1, 3], 'Qcat': ['winter', 'winter', 'winter'], 'Tcat': ['rain', 'rain', 'rain']}, {'year': 2013, 'dates':[dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0)], 'P': [7, 4], 'Pcat': ['winter', 'summer'], 'Q': [5, 4], 'Qcat': ['winter', 'summer'], 'Tcat': ['rain', 'snow']}, {'year': 2014, 'dates': [], 'P': [], 'Pcat': [], 'Q': [], 'Qcat': [], 'Tcat': []}])
    def test_split_fluxes_by_hydro_year2(self):
        with self.assertRaises(Exception):
            split_fluxes_by_hydro_year([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, -5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8])
    def test_split_fluxes_by_hydro_year3(self):
        with self.assertRaises(Exception):
            split_fluxes_by_hydro_year([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, -8, 1, 1, 3, 5, 4, 8])
    def test_split_fluxes_by_hydro_year4(self):
        with self.assertRaises(Exception):
            split_fluxes_by_hydro_year([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, np.NaN, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8])
    def test_split_fluxes_by_hydro_year5(self):
        with self.assertRaises(Exception):
            split_fluxes_by_hydro_year([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2015, 2016], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8])

class TestSplitIsotopesByHydroYear(unittest.TestCase):
    def test_split_isotopes_by_hydro_year1(self):
        self.assertEqual(split_isotopes_by_hydro_year([dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2], 2011), ([[dt(2012, 9, 15, 0, 0), dt(2012, 9, 30, 0, 0)], [dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 4, 30, 0, 0), dt(2013, 5, 15, 0, 0)], []], [[-5, -8], [-8, -2, 0.9, -6, -8, np.NaN, -5, -2, -2], []], [['summer', 'summer'], ['winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'summer'], []], [[-10, -3], [-3, -11.45, -9, -6, -36.6, 0, -4, -2, -2], []], [['summer', 'summer'], ['winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'summer'], []], [[30, 16], [15, 30, 31, 31, 30, 31, 30, 16, 15], []]))
    def test_split_isotopes_by_hydro_year2(self):
        with self.assertRaises(Exception):
            split_isotopes_by_hydro_year([dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2])
    def test_split_isotopes_by_hydro_year3(self):
        with self.assertRaises(Exception):
            split_isotopes_by_hydro_year([dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2])
    def test_split_isotopes_by_hydro_year4(self):
        with self.assertRaises(Exception):
            split_isotopes_by_hydro_year([dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -4, -2])

class TestSumPrecipitationAndRunoff(unittest.TestCase):
    def test_sum_precipitation_and_runoff1(self):
        self.assertEqual(sum_precipitation_and_runoff([[dt(2012, 9, 26, 0, 0), dt(2012, 9, 30, 0, 0)], [dt(2012, 10, 5, 0, 0), dt(2012, 10, 6, 0, 0)]], [{'dates':[dt(2012, 9, 25, 0, 0), dt(2012, 9, 26, 0, 0), dt(2012, 9, 27, 0, 0), dt(2012, 9, 28, 0, 0), dt(2012, 9, 29, 0, 0), dt(2012, 9, 30, 0, 0)], 'P':[2.5, 1, 0, 0, 3, 0.5], 'Q': [2.1, 2.9, 1, 1.8, 2.2, 2]}, {'dates': [dt(2012, 10, 1, 0, 0), dt(2012, 10, 2, 0, 0), dt(2012, 10, 3, 0, 0), dt(2012, 10, 4, 0, 0), dt(2012, 10, 5, 0, 0), dt(2012, 10, 6, 0, 0)], 'P': [0, 1.5, 0, 0, 0, 0], 'Q': [1, 1.1, 1, 0.9, 0.8, 0.7]}], [[-12, -11], [-11, -11.5]], [[-10, -10.2], [-10.1, -10]]), ([[3.5, 3.5], [1.5, 0]], [[2.9, 2], [0.8, 0.7]]))
    def test_sum_precipitation_and_runoff2(self):
        self.assertEqual(sum_precipitation_and_runoff([[dt(2012, 9, 26, 0, 0), dt(2012, 9, 30, 0, 0)], [dt(2012, 10, 5, 0, 0), dt(2012, 10, 6, 0, 0)]], [{'dates':[dt(2012, 9, 25, 0, 0), dt(2012, 9, 26, 0, 0), dt(2012, 9, 27, 0, 0), dt(2012, 9, 28, 0, 0), dt(2012, 9, 29, 0, 0), dt(2012, 9, 30, 0, 0)], 'P':[2.5, 1, 0, 0, 3, 0.5], 'Q': [2.1, 2.9, 1, 1.8, 2.2, 2]}, {'dates': [dt(2012, 10, 1, 0, 0), dt(2012, 10, 2, 0, 0), dt(2012, 10, 3, 0, 0), dt(2012, 10, 4, 0, 0), dt(2012, 10, 5, 0, 0), dt(2012, 10, 6, 0, 0)], 'P': [0, 1.5, 0, 0, 0, 0], 'Q': [1, 1.1, 1, 0.9, 0.8, 0.7]}], [[-12, -11], [-11, np.nan]], [[np.nan, -10.2], [-10.1, -10]]), ([[3.5, 3.5], [1.5]], [[2], [0.8, 0.7]]))

class TestRemoveNanSamples(unittest.TestCase):
    def test_remove_nan_samples1(self):
        self.assertEqual(remove_nan_samples([2007, 2008], [[4, np.nan, 5, 6, np.nan], [np.nan, 1, np.nan]], [['winter', 'summer', 'winter', 'summer', 'summer'], ['winter', 'summer', 'winter']]), ([[4, 5, 6], [1]], [['winter', 'winter', 'summer'], ['summer']]))
    def test_remove_nan_samples2(self):
        self.assertEqual(remove_nan_samples([2002], [[3, 8]], [['winter', 'summer']]), ([[3, 8]], [['winter', 'summer']]))

class TestWtdMean(unittest.TestCase):
    def test_wtd_mean1(self):
        self.assertEqual(wtd_mean([1, 4, 7], [1, 1, 1]), (4, 1.7320508075688772))
    def test_wtd_mean2(self):
        self.assertEqual(wtd_mean([1, 4, 7]), (4, 1.7320508075688772))
    def test_wtd_mean3(self):
        with self.assertRaises(Exception):
            wtd_mean([1, 2], [3, -2])

class TestDataChecks(unittest.TestCase):
    def test_data_checks1(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'sum', 'sum'], ['winter', 'summer'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: Pcat and Pdelcat use different labels', str(cm.exception))
    def test_data_checks2(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'summer', 'summer'], ['winter', 'summer'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['win', 'win', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: Qcat and Qdelcat use different labels', str(cm.exception))
    def test_data_checks3(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'winter', 'winter'], ['winter', 'winter'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: need exactly two precipitation categories', str(cm.exception))
    def test_data_checks4(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'summer', 'summer'], ['winter', 'summer'], [-12, -10, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: Pdel, Pwt, and Pdelcat must have the same length', str(cm.exception))
    def test_data_checks5(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'summer', 'summer'], ['winter', 'summer'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: Qdel, Qwt, and Qdelcat must have the same length', str(cm.exception))
    def test_data_checks6(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'summer'], ['winter', 'summer'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2, 2])
        self.assertEqual('fatal error: P and Pcat must have the same length', str(cm.exception))
    def test_data_checks7(self):
        with self.assertRaises(Exception) as cm:
            data_checks(['winter', 'winter', 'summer', 'summer'], ['winter', 'summer'], [-12, -9], [8, 4],
                    [4, 4, 3, 1], ['winter', 'winter', 'summer', 'summer', 'summer'], ['winter', 'summer', 'summer'],
                    [-11.5, -8, -7], [6, 3, 2], [3, 3, 1, 2])
        self.assertEqual('fatal error: Q and Qcat must have the same length', str(cm.exception))

class TestCalcIsotopesAndFluxes(unittest.TestCase):
    def test_calc_isotopes_and_fluxes1(self):
        self.assertEqual(calc_isotopes_and_fluxes([-12, -13, -10, -15], ['winter', 'winter', 'summer', 'summer'], [0.5, 3, 1, 2], [0.1, 0.2, 0.2, 1, 2, 0.5, 0.5, 1, 1], ['winter', 'winter', 'winter', 'winter', 'winter', 'summer', 'summer', 'summer', 'summer']), ([-13.333333333333334, -12.857142857142858], [2.635231383473649, 0.614451804788759], [3.0, 3.5], [0.5773502691896257, 1.8165902124584952], 6.5, 1.9061304607327731, -13.076923076923077, 2.1784856354741695))
    def test_calc_isotopes_and_fluxes2(self):
        with self.assertRaises(Exception):
            calc_isotopes_and_fluxes([-12, -13, -10, -15], ['winter', 'winter', 'summer', 'summer'], [np.nan, 3, 1, 2], [0.1, 0.2, 0.2, 1, 2, 0.5, 0.5, 1, 1], ['winter', 'winter', 'winter', 'winter', 'winter', 'summer', 'summer', 'summer', 'summer'])

class TestCalcEtValues(unittest.TestCase):
    def test_calc_et_values1(self):
        self.assertEqual(calc_et_values(1000, 800, 10, 5, -10, -10.5, [-8, -12], [400, 400], [1, 0.5], 0.5, [10, 20]), (200, 11.180339887498949, -8.0, 3.027194451963732))
    def test_calc_et_values2(self):
        with self.assertRaises(Exception):
            calc_et_values(np.nan, 800, 10, 5, -10, -10.5, [-8, -12], [400, 400], [1, 0.5], 0.5, [10, 20])

class TestEndMemberMixing(unittest.TestCase):
    def test_end_member_mixing1(self):
        self.assertEqual(end_member_mixing([-8, -12], [-9, -11, -10, -8], [0.5, 1], [1, 2, 1.5, 2.5], [500, 500], [400, 400, 800, 200], [15, 12], [12, 10, 15, 5], 800, 200), ([[.75, .25], [.25, .75], [0.5, 0.5], [0.5, 0.5]], [[0.27421763710600383, 0.27421763710600383], [0.5349138365194903, 0.5349138365194903], [0.40019526483955303, 0.40019526483955303], [1.6015012488287357, 1.6015012488287357]]))
    def test_end_member_mixing2(self):
        self.assertEqual(end_member_mixing([-10, -11], [-10, -10, -10, -10], [1, 1], [1, 1, 1, 1], [500, 500], [400, 400, 800, 200], [0, 0], [0, 0, 0, 0], 800, 200), ([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.5, 2.5]], [[1.4142135623730951, 1.4142135623730951], [1.4142135623730951, 1.4142135623730951], [1.4142135623730951, 1.4142135623730951], [5.656854249492381, 5.656854249492381]]))

class TestEndMemberSplitting(unittest.TestCase):
    def test_end_member_splitting1(self):
        self.assertEqual(end_member_splitting([500, 400, 900, 200], [500, 600], [[0.25, 0.75], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]], [[.1, .1], [.2, .2], [.1, .1], [.3, .3]], [10, 15, 5, 20], [8, 10]), ([[.25, 0.625], [.24, 0.4666666666666667], [1.62, 0.15], [-0.6200000000000001, 0.85]], [[0.10020479030465561, 0.08490701613464513], [0.16029892576059268, 0.13470160208749662], [0.1820792311055822, 0.15002314636230119], [0.1820792311055822, 0.15002314636230119]]))

class TestFormatTables(unittest.TestCase):
    def test_format_tables1(self):
        pd.testing.assert_frame_equal(format_tables([[0., 0.], [0., 0.], [0., 0.], [0., 0.]], [[1., 1.], [1., 1.], [1., 1.], [1., 1.]], [[2., 2.], [2., 2.], [2., 2.], [2., 2.]], [[0., 0], [0., 0.], [0., 0.], [0., 0.]]), (pd.DataFrame(data=[[0., 0., 1., 1., 2., 2., 0., 0.], [0., 0., 1., 1., 2., 2., 0., 0.], [0., 0., 1., 1., 2., 2., 0., 0.], [0., 0., 1., 1., 2., 2., 0., 0.]], columns=['f.summer', 'f.winter', 'f.summer.se', 'f.winter.se', 'eta.summer', 'eta.winter', 'eta.summer.se', 'eta.winter.se'], index=['summer', 'winter', 'AllQ', 'ET'])))

class TestEndsplit(unittest.TestCase):
    def test_endsplit1(self):
        self.assertEqual(endsplit([-11, -11, -11, -9, -9, -9], [-10, -10, -10, -10, -10, -10], [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1], ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'],
                        ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'], [1, 1, 1, 1, 1, 1],
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'],
                        ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'])[0],
                         ([0, 3, -10, 6, 3, 0, -9, -11, .5, 0, 3, 0, -10, 0.5, 0, 3, 0, 1, 0]))
    def test_endsplit2(self):
        pd.testing.assert_frame_equal(endsplit([-11, -11, -11, -9, -9, -9], [-10, -10, -10, -10, -10, -10], [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1], ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'],
                        ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'], [1, 1, 1, 1, 1, 1],
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'],
                        ['winter', 'winter', 'winter', 'summer', 'summer', 'summer'])[1],
                        (pd.DataFrame([[0.5, 0.5, 0., 0., 0.25, 0.25, 0., 0.], [0.5, 0.5, 0., 0., 0.25, 0.25, 0., 0.],
                        [0.5, 0.5, 0., 0., 0.5, 0.5, 0., 0.], [0.5, 0.5, 0., 0., 0.5, 0.5, 0., 0.]], index=['summer', 'winter', 'AllQ', 'ET'],
                        columns=['f.summer', 'f.winter', 'f.summer.se', 'f.winter.se', 'eta.summer', 'eta.winter', 'eta.summer.se', 'eta.winter.se'])))



class TestToYearFraction(unittest.TestCase):
    def test_to_year_fraction1(self):
        self.assertEqual(to_year_fraction(dt(2012, 1, 1, 0, 0)), (0.))
    def test_to_year_fraction2(self):
        self.assertEqual(to_year_fraction(dt(2012, 2, 6, 14, 24)), (0.1))

class TestCalcPrecip(unittest.TestCase):
    def test_calc_precip1(self):
        self.assertEqual(calc_precip(pd.Series(['1/1/2013', '1/1/2013', '1/2/2013', '2/1/2013', '2/2/2013', '3/1/2013', '3/2/2013', '4/1/2013', '4/2/2013', '5/1/2013', '5/2/2013', '6/1/2013', '6/2/2013', '7/1/2013', '7/2/2013', '8/1/2013', '8/2/2013', '9/1/2013', '9/2/2013', '10/1/2013', '10/2/2013', '11/1/2013', '11/2/2013', '12/1/2013', '12/2/2013']), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series([-13, -13, -13, -12, -12, -12, -12, -10, -10, -9, -9, -8, -8, -6, -6, -7, -7, -9, -9, -10, -10, -10, -10, -11, -11]), pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4, 9), (-11., 0.36313651960128146, -8., 0.4714045207910317, [-13., -12., -12., -10., -9., -8., -6., -7., -9., -10., -10., -11.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

class TestCalcQ(unittest.TestCase):
    def test_calc_q1(self):
        self.assertEqual(calc_q([2, 3, 2, 5], pd.Series([-10, -10]), pd.Series(['1/3/2014', '1/7/2014']), ['1/3/2014', '1/4/2013', '1/7/2014', '1/9/2015']), (-10, 0))
    def test_calc_q2(self):
        self.assertEqual(calc_q([2, 3, 2, 5], pd.Series([-10, -13]), pd.Series(['1/3/2014', '1/7/2014']), ['1/3/2014', '1/4/2013', '1/7/2014', '1/9/2015']), (-11.5, 1.224744871391589))

class TestCalculateWLS(unittest.TestCase):
    def test_calculate_wls1(self):
        self.assertEqual(calculate_wls([750, 705, 796], [2, 1.1, 2.9], [1, 1, 1], x_bound='P_s'),
                         ({'xlabel':(705, 750, 796), 'f':(1.1, 2, 2.9),
                            'x_val_ci': range(700, 801, 25), 'fitvals':[1.1033327966511006, 1.9934068587989042, 2.9032603445499934],
                          'ci_low':[-0.19613011876328956, 0.30036015710365915, 0.795526512745637, 1.2893666921907507, 1.7818828343749764], 'ci_upp':[2.205001476032646, 2.697482380329925, 3.1912872048521717, 3.6864182055712855, 4.182873243551287],
                         'slope':0.019779423603284538, 'slope p-val':0.004038985334008691}))


class TestUndercatchCorrection(unittest.TestCase):
    def test_undercatch_correction1(self):
        pd.testing.assert_frame_equal(undercatch_correction(pd.DataFrame({'Year': [1994, 1995], 'Ptot': [1200, 1100],
                    'P_s': [600, 600], 'P_w': [600, 500]}), pd.DataFrame({'Year': [1994, 1995], 'Undercatch (mm)': [200, 100]})),
                    (pd.DataFrame({'Year': [1994, 1995], 'Ptot': [1400, 1200], 'P_s': [600, 600], 'P_w': [800, 600], 'ratio':[0.75, 1.]})))
    def test_undercatch_correction2(self):
        pd.testing.assert_frame_equal(undercatch_correction(pd.DataFrame({'Year': [1994, 1996], 'Ptot': [1200, 1100], 'P_s': [600, 600], 'P_w': [600, 500]}),
                pd.DataFrame({'Year': [1994, 1995], 'Undercatch (mm)': [200, 100]})), (pd.DataFrame({'Year': [1994, 1996],
                                                'Ptot': [float(1400), 1250], 'P_s': [600, 600], 'P_w': [800, float(650)], 'ratio': [.75, float(600/650)]})))

class TestCalculateFractions(unittest.TestCase):
    def test_calculate_fractions(self):
        pd.testing.assert_frame_equal(calculate_fractions(pd.DataFrame({'Year': [1991, 1992, 1993, 1994], 'P_s': [400, 400, 500, 500],
                                'Ptot': [800, 800, 1000, 1000], 'Q': [600, 600, 750, 750], 'Pdel_s': [-9, -9, -8, -8],
                                'Pdel_w': [-11, -11, -12, -12], 'Qdel': [-10, -10, -10, -10], 'ET': [100, 100, 150, 150],
                                'f_ET': [.25, .25, .5, .25], 'f_ET_se': [0.1, 0.1, 0.2, 0.1],
                                'f_Ps':[.5, 0.5, .5, .75], 'f_Ps_se': [.1, .2, .1, .1]}), et='mass bal'),
                         (pd.DataFrame({'Year': [1991, 1992, 1993, 1994], 'P_s': (400, 400, 500, 500), 'Ptot': [800, 800, 1000, 1000],
                           'Q': [600, 600, 750, 750], 'Pdel_s': [-9, -9, -8, -8], 'Pdel_w': [-11, -11, -12, -12],
                           'Qdel': [-10, -10, -10, -10], 'ET': [200, 200, 250, 250], 'f_ET': [0.5, 0.5, 0.5, 0.5],
                           'f_ET_se': [0.2, 0.2, 0.2, 0.2], 'f_Ps':[.25, 0.25, .25, .25], 'f_Ps_se': [.05, .1, .05, .03333333333333333]})))

class TestRoundHalfUp(unittest.TestCase):
    def test_round_half_up1(self):
        self.assertEqual(round_half_up(5.5), (6))
    def test_round_half_up2(self):
        self.assertEqual(round_half_up(6.4), (6))

class TestPlotPanels(unittest.TestCase):
    def test_plot_panels(self):
        self.assertEqual(plot_panels({'No Lag':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 1,
                 'f_ET_slope_pval': 0.08,
                 'f_Ps_slope': 1,
                 'f_Ps_slope_pval': 1.},
            'Lag 1':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -1,
                 'f_ET_slope_pval': 0.0999,
                 'f_Ps_slope': 2,
                 'f_Ps_slope_pval': 0.99},
            'Lag 2':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 1,
                 'f_ET_slope_pval': 0.01,
                 'f_Ps_slope': -1,
                 'f_Ps_slope_pval': 0.001},
            'Mixed':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 2,
                 'f_ET_slope_pval': .4,
                 'f_Ps_slope': 1,
                 'f_Ps_slope_pval': .6},
            'Lag 1 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -1,
                 'f_ET_slope_pval': 0.02,
                 'f_Ps_slope': -2,
                 'f_Ps_slope_pval': 0.01},
            'Lag 2 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 3,
                 'f_ET_slope_pval': .03,
                 'f_Ps_slope': 4,
                 'f_Ps_slope_pval': .01}},
            {'No Lag':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -3,
                 'f_ET_slope_pval': .01,
                 'f_Ps_slope': 3,
                 'f_Ps_slope_pval': 0.1},
            'Lag 1':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 4,
                 'f_ET_slope_pval': .02,
                 'f_Ps_slope': 3,
                 'f_Ps_slope_pval': .02},
            'Lag 2':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -2,
                 'f_ET_slope_pval': .8,
                 'f_Ps_slope': 2,
                 'f_Ps_slope_pval': .8},
            'Mixed':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -3,
                 'f_ET_slope_pval': .08,
                 'f_Ps_slope': 2,
                 'f_Ps_slope_pval': .8},
            'Lag 1 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 1,
                 'f_ET_slope_pval': .8,
                 'f_Ps_slope': 1,
                 'f_Ps_slope_pval': .8},
            'Lag 2 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': 4,
                 'f_ET_slope_pval': .7,
                 'f_Ps_slope': 4,
                 'f_Ps_slope_pval': .7}},
            {'No Lag':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -6,
                 'f_ET_slope_pval': .09,
                 'f_Ps_slope': -6,
                 'f_Ps_slope_pval': .09},
            'Lag 1':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': -4.,
                 'f_ET_slope_pval': .08,
                 'f_Ps_slope': 3,
                 'f_Ps_slope_pval': .1},
            'Lag 2':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': .1,
                 'f_ET_slope_pval': .1,
                 'f_Ps_slope': .1,
                 'f_Ps_slope_pval': .1},
            'Mixed':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': .1,
                 'f_ET_slope_pval': .1,
                 'f_Ps_slope': .1,
                 'f_Ps_slope_pval': .1},
            'Lag 1 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': .1,
                 'f_ET_slope_pval': .1,
                 'f_Ps_slope': .1,
                 'f_Ps_slope_pval': .1},
            'Lag 2 Mean':{'xlabel': [50, 50, 50],
                 'f_ET': [0.5, 0.5, 0.5],
                 'f_Ps': [0.5, 0.5, 0.5],
                 'x_val_ci': range(25, 76, 25),
                 'f_ET_fitvals': [0.5, 0.5, 0.5],
                 'f_ET_ci_low': [0.5, 0.5, 0.5],
                 'f_ET_ci_upp': [0.5, 0.5, 0.5],
                 'f_Ps_fitvals': [0.5, 0.5, 0.5],
                 'f_Ps_ci_low': [0.5, 0.5, 0.5],
                 'f_Ps_ci_upp': [0.5, 0.5, 0.5],
                 'f_ET_slope': .1,
                 'f_ET_slope_pval': .1,
                 'f_Ps_slope': .1,
                 'f_Ps_slope_pval': .1}}, 'P_s', 'title'), ({'TotalCount': {'+': 4, '-': 6, 'NA': 8},
            'ByWatershedCount': {'All RHB': {'+': 3, '-': 2, 'NA': 1}, 'Upper RHB': {'+': 1, '-': 2, 'NA': 3},
                          'Lysimeter': {'+': 0, '-': 2, 'NA': 4}},
            'ByMethodCount': {'No Lag': {'+': 1, '-': 2, 'NA': 0}, 'Lag 1': {'+': 1, '-': 2, 'NA': 0},
                       'Lag 2': {'+': 1, '-': 0, 'NA': 2},
                       'Mixed': {'+': 0, '-': 1, 'NA': 2}, 'Lag 1 Mean': {'+': 0, '-': 1, 'NA': 2},
                       'Lag 2 Mean': {'+': 1, '-': 0, 'NA': 2}},
            'ByWatershedandMethod': {
            'All RHB': {'No Lag': {'+': 1, '-': 0, 'NA': 0}, 'Lag 1': {'+': 0, '-': 1, 'NA': 0},
                  'Lag 2': {'+': 1, '-': 0, 'NA': 0},
                  'Mixed': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1 Mean': {'+': 0, '-': 1, 'NA': 0},
                  'Lag 2 Mean': {'+': 1, '-': 0, 'NA': 0}},
            'Upper RHB': {'No Lag': {'+': 0, '-': 1, 'NA': 0}, 'Lag 1': {'+': 1, '-': 0, 'NA': 0},
                    'Lag 2': {'+': 0, '-': 0, 'NA': 1},
                    'Mixed': {'+': 0, '-': 1, 'NA': 0}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 1},
                    'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 1}},
            'Lysimeter': {'No Lag': {'+': 0, '-': 1, 'NA': 0}, 'Lag 1': {'+': 0, '-': 1, 'NA': 0},
                    'Lag 2': {'+': 0, '-': 0, 'NA': 1},
                    'Mixed': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 1},
                    'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 1}}}},
            {'TotalCount': {'+': 2, '-': 3, 'NA': 13},
            'ByWatershedCount': {'All RHB': {'+': 1, '-': 2, 'NA': 3}, 'Upper RHB': {'+': 1, '-': 0, 'NA': 5},
                         'Lysimeter': {'+': 0, '-': 1, 'NA': 5}},
            'ByMethodCount': {'No Lag': {'+': 0, '-': 1, 'NA': 2}, 'Lag 1': {'+': 1, '-': 0, 'NA': 2},
                      'Lag 2': {'+': 0, '-': 1, 'NA': 2},
                      'Mixed': {'+': 0, '-': 0, 'NA': 3}, 'Lag 1 Mean': {'+': 0, '-': 1, 'NA': 2},
                      'Lag 2 Mean': {'+': 1, '-': 0, 'NA': 2}},
            'ByWatershedandMethod': {'All RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1': {'+': 0, '-': 0, 'NA': 1},
                      'Lag 2': {'+': 0, '-': 1, 'NA': 0},
                      'Mixed': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1 Mean': {'+': 0, '-': 1, 'NA': 0},
                      'Lag 2 Mean': {'+': 1, '-': 0, 'NA': 0}}, 'Upper RHB': {'No Lag': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1': {'+': 1, '-': 0, 'NA': 0},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 1},
                      'Mixed': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 1},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 1}}, 'Lysimeter': {'No Lag': {'+': 0, '-': 1, 'NA': 0}, 'Lag 1': {'+': 0, '-': 0, 'NA': 1},
                      'Lag 2': {'+': 0, '-': 0, 'NA': 1},
                      'Mixed': {'+': 0, '-': 0, 'NA': 1}, 'Lag 1 Mean': {'+': 0, '-': 0, 'NA': 1},
                      'Lag 2 Mean': {'+': 0, '-': 0, 'NA': 1}}}}))

class TestPlotEtAmounts(unittest.TestCase):
    def test_plot_et_amounts1(self):
        self.assertEqual(plot_et_amounts([2000, 2001, 2002, 2003, 2004, 2005, 2006], pd.DataFrame({'Year': [2000, 2002, 2003, 2004, 2005, 2006], 'ET': [500, 550, 525, 510, 490, 505]}), pd.DataFrame({'Year': [2000, 2001, 2003, 2005, 2006], 'ET': [505, 507, 503, 495, 511]}), pd.DataFrame({'Year': [2003, 2004], 'ET': [500, 501]}), pd.DataFrame({'Year': [2004], 'annual_ET': [506]})), ({"All": [500, 0, 550, 525, 510, 490, 505], "Upper": [505, 507, 0, 503, 0, 495, 511], "Lysimeter discharge": [0, 0, 0, 500, 501, 0, 0], "Lysimeter weights": [0, 0, 0, 0, 506, 0, 0]}))

class TestCalculateAvgEt(unittest.TestCase):
    def test_calculate_avg_et1(self):
        pd.testing.assert_frame_equal(calculate_avg_et(pd.DataFrame({'Ptot': [1000, 1200, 1400], 'Q': [800, 800, 1100]})), (pd.DataFrame({'Ptot': [1000, 1200, 1400], 'Q': [700., 900, 1100], 'ET': [300., 300, 300]})))

class TestCalculateScaledEt(unittest.TestCase):
    def test_calculate_scaled_et1(self):
        pd.testing.assert_frame_equal(calculate_scaled_et(pd.DataFrame({'Year': [2000, 2001, 2002, 2003], 'Ptot': [1000, 1100, 900, 1000], 'Q': [800, 900, 800, 900]}), pd.DataFrame({'Year': [2001, 2002, 2003], 'annual_ET': [120, 180, 150]})), (pd.DataFrame({'Year': [2000, 2001, 2002, 2003], 'Ptot': [1000, 1100, 900, 1000], 'Q': [850., 980, 720, 850], 'ET': [150., 120, 180, 150]})))

class TestWorkflowEndsplit(unittest.TestCase):
    def test_workflow_endsplit1(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['No Lag'], (pd.DataFrame([[2003,
                                2., -10., 4., 2., 0., -8., -12., .5, 0., 2., 0., -10., .5, 0., 2., 0., 1., 0.], [2004, 2., -10., 4., 2.,
                                0., -9., -12., 0.33333333333333337, 0., 2., 0., -10.5, 0.33333333333333337, 0., 2., 0., 1., 0.],
                                [2005, 2., -10., 4., 2., 0., -10., -12., 0., 0., 2., 0., -11., 0., 0., 2., 0., 1., 0.]], columns=['Year',
                                'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se',
                                 'ET', 'ET_se', 'AllP_del', 'f_Ps', 'f_Ps_se', 'P_w', 'P_w_se', 'ratio', 'ratio_se'])))
    def test_workflow_endsplit2(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['Lag 1'], (pd.DataFrame([[2004,
                                2., -10., 4., 2., 0., -8., -12., .5, 0., 2., 0., -10., .5, 0., 2., 0., 1., 0.], [2005, 2., -10., 4., 2.,
                                0., -9., -12., 0.33333333333333337, 0., 2., 0., -10.5, 0.33333333333333337, 0., 2., 0., 1., 0.]], columns=['Year',
                                'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se',
                                 'ET', 'ET_se', 'AllP_del', 'f_Ps', 'f_Ps_se', 'P_w', 'P_w_se', 'ratio', 'ratio_se'])))
    def test_workflow_endsplit3(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['Lag 2'], (pd.DataFrame([[2005,
                                2., -10., 4., 2., 0., -8., -12., .5, 0., 2., 0., -10., .5, 0., 2., 0., 1., 0.]], columns=['Year',
                                'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se',
                                 'ET', 'ET_se', 'AllP_del', 'f_Ps', 'f_Ps_se', 'P_w', 'P_w_se', 'ratio', 'ratio_se'])))
    def test_workflow_endsplit4(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                    '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                    pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10,-10]), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    pd.Series(data=[-10., -10., -10., -10, -10, -10, -10, -10, -10, -10, -10, -10]),
                                    ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                     '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['Mixed'],
                                (pd.DataFrame([[2003, 2., -10., 4., 2., 0., -9., -12., 0.33333333333333337, 0.08114408259335794, 2., 0., -10.5, 0.33333333333333337, 0.08114408259335794, 2., 0., 1., 0.],
                                                [2004, 2., -10., 4., 2., 0., -9., -12., 0.33333333333333337, 0.08114408259335794, 2., 0., -10.5, 0.33333333333333337, 0.08114408259335794, 2., 0., 1., 0.],
                                                [2005, 2., -10., 4., 2., 0., -9., -12., 0.33333333333333337, 0.08114408259335794, 2., 0., -10.5, 0.33333333333333337, 0.08114408259335794, 2., 0., 1., 0.]], columns=['Year',
                                                                                     'Q', 'Qdel', 'Ptot', 'P_s',
                                                                                     'P_s_se', 'Pdel_s', 'Pdel_w',
                                                                                     'f_ET', 'f_ET_se',
                                                                                     'ET', 'ET_se', 'AllP_del', 'f_Ps',
                                                                                     'f_Ps_se', 'P_w', 'P_w_se',
                                                                                     'ratio', 'ratio_se'])))
    def test_workflow_endsplit5(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['Lag 1 Mean'], (pd.DataFrame([[2004,
                                2., -10., 4., 2., 0., -8.5, -12., 0.4285714285714286, 0.047130634219561277, 2., 0., -10.25, 0.4285714285714286, 0.047130634219561277, 2., 0., 1., 0.],
                                [2005, 2., -10., 4., 2., 0., -9.5, -12., 0.19999999999999996, 0.09237604307034011, 2., 0., -10.75, 0.19999999999999996, 0.09237604307034011, 2., 0., 1., 0.]], columns=['Year',
                                'Q', 'Qdel', 'Ptot', 'P_s', 'P_s_se', 'Pdel_s', 'Pdel_w', 'f_ET', 'f_ET_se',
                                 'ET', 'ET_se', 'AllP_del', 'f_Ps', 'f_Ps_se', 'P_w', 'P_w_se', 'ratio', 'ratio_se'])))
    def test_workflow_endsplit6(self):
        pd.testing.assert_frame_equal(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                    '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                    pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10,-10]), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    pd.Series(data=[-10., -10., -10., -10, -10, -10, -10, -10, -10, -10, -10, -10]),
                                    ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                     '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[0]['Lag 2 Mean'],
                                (pd.DataFrame([[2005, 2., -10., 4., 2., 0., -9., -12., 0.33333333333333337, 0.08114408259335794, 2., 0., -10.5, 0.33333333333333337, 0.08114408259335794, 2., 0., 1., 0.]], columns=['Year',
                                                                                     'Q', 'Qdel', 'Ptot', 'P_s',
                                                                                     'P_s_se', 'Pdel_s', 'Pdel_w',
                                                                                     'f_ET', 'f_ET_se',
                                                                                     'ET', 'ET_se', 'AllP_del', 'f_Ps',
                                                                                     'f_Ps_se', 'P_w', 'P_w_se',
                                                                                     'ratio', 'ratio_se'])))
    def test_workflow_endsplit7(self):
        self.assertEqual(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[1], ({'Year': 3, 'Q': 6., 'Qdel': -10.,
                                'Ptot': 12., 'P_s': 6., 'P_s_se': 0., 'Pdel_s': -9., 'Pdel_w': -12., 'f_ET': 0.33333333333333333,
                                'f_ET_se': 0.08114408259335794, 'ET': 6., 'ET_se': 0., 'AllP_del': -10.5, 'f_Ps': 0.33333333333333337, 'f_Ps_se': 0.08114408259335794,
                                'P_w': 6., 'P_w_se': 0., 'ratio': 1., 'ratio_se': 0.}))
    def test_workflow_endsplit8(self):
        self.assertEqual(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[2], ([.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05]))
    def test_workflow_endsplit9(self):
        self.assertEqual(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[3], ([-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]))
    def test_workflow_endsplit10(self):
        self.assertEqual(workflow_endsplit(['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                pd.Series(data=[-12., -12, -8, -8, -12, -12, -9, -9, -12, -12, -10, -10]),
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pd.Series(data=[-10., -10., -10., -10, -10, -10, -10,
                                -10, -10, -10, -10, -10]), ['10/15/03', '1/15/04', '5/15/04', '7/15/04', '11/15/04',
                                '2/15/05', '5/15/05', '6/15/05', '12/15/05', '3/15/06', '6/15/06', '9/15/06'],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Upper")[4], ([dt(2003, 10, 15, 0, 0), dt(2004, 1, 15, 0, 0),
                                dt(2004, 5, 15, 0, 0), dt(2004, 7, 15, 0, 0), dt(2004, 11, 15, 0, 0), dt(2005, 2, 15, 0, 0),
                                dt(2005, 5, 15, 0, 0), dt(2005, 6, 15, 0, 0), dt(2005, 12, 15, 0, 0), dt(2006, 3, 15, 0, 0), dt(2006, 6, 15, 0, 0), dt(2006, 9, 15, 0, 0)]))

class TestConfidenceIntervals(unittest.TestCase):
    maxDiff = None
    def test_confidence_intervals1(self):
        self.assertEqual(confidence_intervals({"No Lag": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]}), "Lag 1": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]}), "Lag 2": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]}), "Mixed": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]}), "Lag 1 Mean": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]}),
            "Lag 2 Mean": pd.DataFrame({'P_s': [500, 530, 530, 550], 'f_ET': [.5, .6, .7, .8],
                'f_ET_se': [1, 2, 1, 1], 'f_Ps': [.5, .5, .5, .5],
                'f_Ps_se': [1, 2, 1, 1]})}, xlabel='P_s'),
            ({"No Lag": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1},
            "Lag 1": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1},
                  "Lag 2": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1},
            "Mixed": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1},
             "Lag 1 Mean": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1},
            "Lag 2 Mean": {'xlabel': (500, 530, 530, 550), 'f_ET': (0.5, 0.6, 0.7, 0.8), 'f_Ps': (.5, .5, .5, .5), 'x_val_ci': range(495, 555, 25),
              'f_ET_fitvals': [0.5, 0.6799999999999997, 0.6799999999999997, 0.7999999999999998], 'f_ET_ci_low': [-1.5474969343051765, -1.3939833085863964, -1.2449927984566411], 'f_ET_ci_upp':
                 [2.487496934305177, 2.6339833085863966, 2.7849927984566403], 'f_Ps_fitvals': [], 'f_Ps_ci_low': [], 'f_Ps_ci_upp': [],
              'f_ET_slope': 0.0059999999999999915, 'f_ET_slope_pval': 0.021192298697526465, 'f_Ps_slope': 0, 'f_Ps_slope_pval': 1}}))



