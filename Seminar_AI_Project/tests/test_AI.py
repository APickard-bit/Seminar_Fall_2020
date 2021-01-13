import unittest
from src.csv_reader import read_csv_file
from src.stats_calculator import calc_sample_mean, calc_std_dev, calc_linear_reg, calc_correlation_coefficiant, calc_sample_median, calc_sample_mode


class TestAI(unittest.TestCase):

    def test_csv_reader(self):
        print("\nTest csv file read-in correctly")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        self.assertEqual(set1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "set1 must be equal to the first set of data values in data1.csv, but equals "+str(set1))
        self.assertEqual(set2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "set2 must be equal to the second set of data values in data1.csv, but equals "+str(set2))

    def test_calulate_mean(self):
        print("\nTest calculate mean in a set of values")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        mean1 = calc_sample_mean(set1)
        mean2 = calc_sample_mean(set2)
        self.assertEqual(5.5, mean1, "Mean of set1 must be 5.5 but got value "+ str(mean1))
        self.assertEqual(11, mean2, "Mean of set2 must be 11 but got value "+ str(mean2))

    def test_calculate_median(self):
        print("\nTest calculate median in a set of values")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        median1 = calc_sample_median(set1)
        median2 = calc_sample_median(set2)
        self.assertEqual(5.5, median1, "Median should also be 5.5, but got value "+str(median1))
        self.assertEqual(11, median2, "Median should also be 11, but got value "+str(median2))

    def test_calculate_nonexistant_mode(self):
        print("\nTest calculate mode in a set of basic values with no acutal mode")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        mode1 = calc_sample_mode(set1)
        mode2 = calc_sample_mode(set2)
        self.assertEqual(0, mode1, "Mode shouldn't acutally exist, but got "+str(mode1))
        self.assertEqual(0, mode2, "Mode shouldn't actually exist, but got "+str(mode2))

    def test_calculate_std_dev(self):
        print("\nTest calculate standard deviation of a set of values")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        mean1 = calc_sample_mean(set1)
        mean2 = calc_sample_mean(set2)
        stddev1 = calc_std_dev(set1, mean1)
        stddev2 = calc_std_dev(set2, mean2)
        self.assertEqual(2.872, stddev1, "Standard Dev of set1 should be 2.872, but got "+str(stddev1))
        self.assertEqual(5.745, stddev2, "Standard Dev of set2 should be 5.745, but got "+str(stddev2))

    def test_calculate_correlation_constant(self):
        print("\nTest the calculation of a correlation constant for linear regression")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        mean1 = calc_sample_mean(set1)
        mean2 = calc_sample_mean(set2)
        stddev1 = calc_std_dev(set1, mean1)
        stddev2 = calc_std_dev(set2, mean2)
        r = calc_correlation_coefficiant(stddev1, stddev2, mean1, mean2, set1, set2)
        self.assertEqual(1, r, "Correlation Corefficient should be 1, but got "+str(r))

    def test_calculate_linear_regression_equation(self):
        print("\nTest the calculation of the slope and y-intercept of a linear regression equation")
        set1, set2 = read_csv_file("tests/test_data/data1.csv")
        mean1 = calc_sample_mean(set1)
        mean2 = calc_sample_mean(set2)
        stddev1 = calc_std_dev(set1, mean1)
        stddev2 = calc_std_dev(set2, mean2)
        r = calc_correlation_coefficiant(stddev1, stddev2, mean1, mean2, set1, set2)
        m, b = calc_linear_reg(stddev1, stddev2, mean1, mean2, r)
        self.assertEqual(2, m, "Equation of the line should be y = 2x")
        self.assertEqual(0, b, "Equation of the line should be y = 2x")