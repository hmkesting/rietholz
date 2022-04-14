import pandas as pd
import math
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from preprocessing import ConvertDateTime, ListYears, FindStartOfSampling, ListUnusableDates, SplitFluxesByHydrologicYear, SplitIsotopesByHydrologicYear
from cleaning import SumPrecipitationAndRunoff, RemoveNanSamples
from calculations import EndSplit
from plot import PlotFraction, PlotAmount
from MainEndSplit import LysCorrectedEndsplit
import unittest
from datetime import datetime as dt

class Test_ConvertDateTime(unittest.TestCase):
    def test_ConvertDateTime1(self):
        self.assertEqual(ConvertDateTime(['08/23/21', '09/24/21']), [dt(2021, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_ConvertDateTime2(self):
        self.assertEqual(ConvertDateTime(['08/23/2021', '09/24/2021']), [dt(2021, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_ConvertDateTime3(self):
        self.assertEqual(ConvertDateTime(['08/23/73', '09/24/21']), [dt(1973, 8, 23, 0, 0), dt(2021, 9, 24, 0, 0)])
    def test_ConvertDateTime4(self):
        self.assertRaises(Exception, ConvertDateTime, ['08/23/21', '24/09/21'])
    def test_ConvertDateTime5(self):
        self.assertRaises(Exception, ConvertDateTime, ['08/23/221', '09/24/021'])
    def test_ConvertDateTime6(self):
        self.assertRaises(ValueError, ConvertDateTime, ['hello', '09/24/021'])
    def test_ConvertDateTime7(self):
        self.assertRaises(Exception, ConvertDateTime, ['', '09/24/021'])

class Test_ListYears(unittest.TestCase):
    def test_ListYears1(self):
        self.assertEqual(ListYears((dt(2019, 3, 14, 0, 0), dt(2003, 10, 27, 0, 0))), [2019, 2003])
    def test_ListYears2(self):
        self.assertEqual(ListYears([dt(2003, 3, 14, 0, 0), dt(2023, 10, 27, 0, 0)]), [2003, 2023])
    def test_ListYears3(self):
        self.assertRaises(Exception, ListYears, ('04/13/2021', ''))
    def test_ListYears4(self):
        self.assertRaises(Exception, ListYears, ('07/11/2014', '06/26'))
    def test_ListYears5(self):
        self.assertRaises(Exception, ListYears, ['07/11/94', '06/26/2006'])
    def test_ListYears6(self):
        self.assertRaises(Exception, ListYears, ('27/11/1994', '06/26/2006'))
    def test_ListYears7(self):
        self.assertRaises(Exception, ListYears, ('hello', '06/26,2006'))
    def test_ListYears8(self):
        self.assertRaises(Exception, ListYears, (568, '06/26,2006'))

class Test_FindStartOfSampling(unittest.TestCase):
    def test_FindStartOfSampling1(self):
        self.assertEqual(FindStartOfSampling([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], [dt(1992, 4, 14, 0, 0), dt(2008, 10, 27, 0, 0)], [-12.456, -15], [456, 586], [-11, -12.0], [434, 632], side='start'), (4, 1993))
    def test_FindStartOfSampling2(self):
        self.assertEqual(FindStartOfSampling([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], [dt(1992, 4, 14, 0, 0), dt(2008, 10, 27, 0, 0)], [np.NaN, 15], [456, 586], [-11, -12.0], [434, 632], side='start'), (10, 2003))
    def test_FindStartOfSampling3(self):
        self.assertEqual(FindStartOfSampling([dt(2003, 10, 27, 0, 0), dt(1993, 4, 14, 0, 0)], [dt(2008, 10, 27, 0, 0), dt(1992, 4, 14, 0, 0)], [-12.456, -15], [456, 586], [-11, -12.0], [434, 632], side='end'), (10, 2003))
    def test_FindStartOfSampling4(self):
        self.assertEqual(FindStartOfSampling([dt(2003, 10, 27, 0, 0), dt(1993, 4, 14, 0, 0)], [dt(2008, 10, 27, 0, 0), dt(1992, 4, 14, 0, 0)], [-9, 15], [np.NAN, 456], [-11, -12.0], [434, 632], side='end'), (4, 1992))
    def test_FindStarOfSampling5(self):
        self.assertRaises(TypeError, FindStartOfSampling, ([dt(1993, 4, 14, 0, 0), dt(2003, 10, 27, 0, 0)], ['hello', -15], [-13, -12.0], [93, 3]), (4, 0))

class Test_ListUnusableDates(unittest.TestCase):
    def test_ListUnusableDates1(self):
        self.assertEqual(ListUnusableDates([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0),  dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], 6, 0, 12, 2, [2012, 2013, 2014]), [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)])
    def test_ListUnusableDates2(self):
        self.assertEqual(ListUnusableDates([dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], 11, 0, 12, 3, [2011, 2012, 2013, 2014]), [dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)])
    def test_ListUnusableDates3(self):
        self.assertEqual(ListUnusableDates([dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2013, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0), dt(2015, 1, 19, 0, 0)], 11, 0, 1, 4, [2011, 2012, 2013, 2014, 2015]), [dt(2011, 11, 2, 0, 0), dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0), dt(2015, 1, 19, 0, 0)])

class Test_SplitFluxesByHydrologicYear(unittest.TestCase):
    def test_SplitFluxesByHydrologicYear1(self):
        self.assertEqual(SplitFluxesByHydrologicYear([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8], [-1, 2, 4, 8, 15, 22, -2, 0]), ([['year': 2012,
    def test_SplitFluxesByHydrologicYear2(self):
        self.assertRaises(Exception, SplitFluxesByHydrologicYear, ([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, -5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8]))
    def test_SplitFluxesByHydrologicYear3(self):
        self.assertRaises(Exception, SplitFluxesByHydrologicYear, ([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, -8, 1, 1, 3, 5, 4, 8]))
    def test_SplitFluxesByHydrologicYear4(self):
        self.assertRaises(Exception, SplitFluxesByHydrologicYear, ([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2013, 2014], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, np.NaN, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8]))
    def test_SplitFluxesByHydrologicYear5(self):
        self.assertRaises(Exception, SplitFluxesByHydrologicYear, ([dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2012, 10, 17, 0, 0), dt(2013, 1, 20, 0, 0), dt(2013, 4, 5, 0, 0), dt(2014, 2, 14, 0, 0), dt(2014, 9, 13, 0, 0), dt(2014, 11, 7, 0, 0)], [2012, 2015, 2016], [dt(2012, 7, 28, 0, 0), dt(2012, 8, 15, 0, 0), dt(2014, 11, 7, 0, 0)], [8, 8, 0.5, 0.2, 5, 7, 4, 8], [8, 8, 1, 1, 3, 5, 4, 8]))

class Test_SplitIsotopesByHydrologicYear(unittest.TestCase):
    def test_SplitIsotopesByHydrologicYear1(self):
        self.assertEqual(SplitIsotopesByHydrologicYear([dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2]), ([[dt(2012, 9, 15, 0, 0), dt(2012, 9, 30, 0, 0)], [dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 4, 30, 0, 0), dt(2013, 5, 15, 0, 0)], []], [[-5, -8], [-8, -2, 0.9, -6, -8, np.NaN, -5, -2, -2], []], [['summer', 'summer'], ['winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'summer'], []], [[-10, -3], [-3, -11.45, -9, -6, -36.6, 0, -4, -2, -2], []], [['summer', 'summer'], ['winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'winter', 'summer'], []]))
    def test_SplitIsotopesByHydrologicYear2(self):
        self.assertRaises(Exception, SplitIsotopesByHydrologicYear, [dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2])
    def test_SplitIsotopesByHydrologicYear3(self):
        self.assertRaises(Exception, SplitIsotopesByHydrologicYear, [dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -36.6, 0, -4, -2])
    def test_SplitIsotopesByHydrologicYear4(self):
        self.assertRaises(Exception, SplitIsotopesByHydrologicYear, [dt(2012, 9, 15, 0, 0), dt(2012, 10, 15, 0, 0), dt(2012, 11, 15, 0, 0), dt(2012, 12, 15, 0, 0), dt(2013, 1, 15, 0, 0), dt(2013, 2, 15, 0, 0), dt(2013, 3, 15, 0, 0), dt(2013, 4, 15, 0, 0), dt(2013, 5, 15, 0, 0)], [30, 31, 30, 31, 31, 30, 31, 30, 31], [2011, 2012, 2013], [], [-5, -8, -2, 0.9, -6, -8, np.NaN, -5, -2], [-10, -3, -11.45, -9, -6, -4, -2])

class Test_LysCorrectedEndsplit(unittest.TestCase):
    def test_LysCorrectedEndsplit1(self):
        self.assertEqual(LysCorrectedEndsplit([1300, 1200], [600, 500], [300, 200], [1990, 1991], [500, 350, 175], [1989, 1990, 1991], [-10.2, -10.5], [[-8.3, -12.4], [-9.1, -11.0]], "Mixed", "All"), [0.13550136, 0.45614035])



