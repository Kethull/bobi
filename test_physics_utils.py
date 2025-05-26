import unittest
import numpy as np
from physics_utils import safe_divide, normalize_vector, PhysicsError

class TestSafeDivide(unittest.TestCase):

    def test_typical_division_scalar(self):
        self.assertAlmostEqual(safe_divide(10, 2), 5.0)
        self.assertAlmostEqual(safe_divide(7, 3), 7/3)
        self.assertAlmostEqual(safe_divide(-10, 2), -5.0)
        self.assertAlmostEqual(safe_divide(10, -2), -5.0)
        self.assertAlmostEqual(safe_divide(0, 5), 0.0)

    def test_division_by_zero_scalar_default_zero(self):
        self.assertAlmostEqual(safe_divide(5, 0), 0.0)
        self.assertAlmostEqual(safe_divide(5, 1e-13), 0.0) # Denominator smaller than default epsilon
        self.assertAlmostEqual(safe_divide(0, 0), 0.0)
        self.assertAlmostEqual(safe_divide(-5, 0), 0.0)

    def test_division_by_zero_scalar_custom_default(self):
        self.assertAlmostEqual(safe_divide(5, 0, default_on_zero_denom=99.0), 99.0)
        self.assertAlmostEqual(safe_divide(0, 0, default_on_zero_denom=99.0), 99.0) # 0/0 with custom default

    def test_division_by_zero_scalar_default_inf(self):
        self.assertEqual(safe_divide(5, 0, default_on_zero_denom=float('inf')), float('inf'))
        self.assertEqual(safe_divide(-5, 0, default_on_zero_denom=float('inf')), float('-inf'))
        self.assertEqual(safe_divide(0, 0, default_on_zero_denom=float('inf')), 0.0) # 0/0 should be 0

    def test_division_by_zero_scalar_default_neg_inf(self):
        # Note: The logic for default_on_zero_denom=float('-inf') is symmetric to float('inf')
        self.assertEqual(safe_divide(5, 0, default_on_zero_denom=float('-inf')), float('inf')) # Numerator positive
        self.assertEqual(safe_divide(-5, 0, default_on_zero_denom=float('-inf')), float('-inf')) # Numerator negative
        self.assertEqual(safe_divide(0, 0, default_on_zero_denom=float('-inf')), 0.0)

    def test_typical_division_numpy_array(self):
        num = np.array([10.0, 7.0, 0.0, -4.0])
        den = np.array([2.0, 3.0, 5.0, -2.0])
        expected = np.array([5.0, 7/3, 0.0, 2.0])
        np.testing.assert_array_almost_equal(safe_divide(num, den), expected)

    def test_division_by_zero_numpy_array_default_zero(self):
        num = np.array([5.0, 0.0, -5.0, 1.0])
        den = np.array([0.0, 0.0, 1e-14, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(safe_divide(num, den), expected)

    def test_division_by_zero_numpy_array_custom_default(self):
        num = np.array([5.0, 0.0])
        den = np.array([0.0, 1e-14])
        expected = np.array([99.0, 99.0])
        np.testing.assert_array_almost_equal(safe_divide(num, den, default_on_zero_denom=99.0), expected)

    def test_division_by_zero_numpy_array_default_inf(self):
        num = np.array([5.0, -5.0, 0.0, 1.0, 0.0])
        den = np.array([0.0, 0.0, 0.0, 1e-15, 1e-16])
        expected = np.array([float('inf'), float('-inf'), 0.0, float('inf'), 0.0])
        result = safe_divide(num, den, default_on_zero_denom=float('inf'))
        np.testing.assert_array_almost_equal(result, expected)

    def test_division_by_zero_numpy_array_all_zeros_denom_default_inf(self):
        num = np.array([1.0, -1.0, 0.0])
        den = np.array([0.0, 0.0, 0.0])
        expected = np.array([float('inf'), float('-inf'), 0.0])
        result = safe_divide(num, den, default_on_zero_denom=float('inf'))
        np.testing.assert_array_almost_equal(result, expected)

class TestNormalizeVector(unittest.TestCase):

    def test_normalize_typical_vector(self):
        vector = np.array([3.0, 4.0])
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

        vector = np.array([1.0, 1.0, 1.0])
        norm = np.sqrt(3)
        expected = np.array([1/norm, 1/norm, 1/norm])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

    def test_normalize_already_normalized_vector(self):
        vector = np.array([0.6, 0.8])
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

    def test_normalize_zero_vector(self):
        vector = np.array([0.0, 0.0, 0.0])
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

    def test_normalize_small_magnitude_vector(self):
        vector = np.array([1e-15, 1e-15]) # Smaller than default epsilon for norm
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

        vector = np.array([1e-10, 1e-10]) # Larger than default epsilon for norm
        norm = np.linalg.norm(vector)
        expected = vector / norm
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)


    def test_normalize_vector_along_axis(self):
        vector = np.array([5.0, 0.0, 0.0])
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

        vector = np.array([0.0, -2.0, 0.0])
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected)

    def test_normalize_list_input(self):
        vector_list = [3, 4]
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalize_vector(vector_list), expected)

    def test_normalize_custom_epsilon(self):
        vector = np.array([1e-5, 1e-5])
        # With default epsilon (1e-12), this vector is normalized
        norm = np.linalg.norm(vector)
        expected_norm = vector / norm
        np.testing.assert_array_almost_equal(normalize_vector(vector), expected_norm)

        # With larger epsilon, this vector is treated as zero
        expected_zero = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(normalize_vector(vector, epsilon=1e-4), expected_zero)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)