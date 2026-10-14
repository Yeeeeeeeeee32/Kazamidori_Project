import math
import unittest
from core.optimization import _hellmann_alpha

class TestHellmannAlpha(unittest.TestCase):
    def test_hellmann_alpha_normal(self):
        # v_lo=5, z_lo=10, v_hi=10, z_hi=100
        # alpha = log(10/5) / log(100/10) = log(2) / log(10) approx 0.301
        v_lo, z_lo, v_hi, z_hi = 5.0, 10.0, 10.0, 100.0
        expected = math.log(2.0) / math.log(10.0)
        self.assertAlmostEqual(_hellmann_alpha(v_lo, z_lo, v_hi, z_hi), expected)

    def test_hellmann_alpha_v_lo_too_small(self):
        # v_lo < 1e-6 triggers fallback to 0.14
        self.assertEqual(_hellmann_alpha(1e-7, 10.0, 10.0, 100.0), 0.14)
        self.assertEqual(_hellmann_alpha(0.0, 10.0, 10.0, 100.0), 0.14)
        self.assertEqual(_hellmann_alpha(-1.0, 10.0, 10.0, 100.0), 0.14)

    def test_hellmann_alpha_z_lo_invalid(self):
        # z_lo <= 0 triggers fallback to 0.14
        self.assertEqual(_hellmann_alpha(5.0, 0.0, 10.0, 100.0), 0.14)
        self.assertEqual(_hellmann_alpha(5.0, -1.0, 10.0, 100.0), 0.14)

    def test_hellmann_alpha_z_hi_not_greater_than_z_lo(self):
        # z_hi <= z_lo triggers fallback to 0.14
        self.assertEqual(_hellmann_alpha(5.0, 10.0, 10.0, 10.0), 0.14)
        self.assertEqual(_hellmann_alpha(5.0, 10.0, 10.0, 5.0), 0.14)

    def test_hellmann_alpha_v_hi_near_zero(self):
        # v_hi is clamped to 1e-9 using max(v_hi, 1e-9)
        v_lo, z_lo, v_hi, z_hi = 5.0, 10.0, 0.0, 100.0
        expected = math.log(1e-9 / 5.0) / math.log(10.0)
        self.assertAlmostEqual(_hellmann_alpha(v_lo, z_lo, v_hi, z_hi), expected)

if __name__ == '__main__':
    unittest.main()
