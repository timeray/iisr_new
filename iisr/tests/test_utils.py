import numpy as np
from unittest import TestCase, main
from iisr.utils import uneven_mean


class TestUnevenMean(TestCase):
    def test_const(self):
        x = np.array([0, 1, 2, 4, 7, 9, 10], dtype=float)
        y = np.ones_like(x) * 3.0
        for method in ['trapz', 'simps']:
            self.assertAlmostEqual(3.0, uneven_mean(x, y, method=method))
            self.assertAlmostEqual(3.0, uneven_mean(x + 20, y, method=method))
            self.assertAlmostEqual(3.0, uneven_mean(x - 2, y, method=method))
            self.assertAlmostEqual(-3.0, uneven_mean(x, -1 * y, method=method))

    def test_even_distribution(self):
        n = 100000
        x = np.arange(n)
        y = np.random.randn(n)
        self.assertAlmostEqual(uneven_mean(x, y), y.mean(), places=4)


if __name__ == '__main__':
    main()
