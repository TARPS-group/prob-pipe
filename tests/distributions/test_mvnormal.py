import unittest
import numpy as np
from probpipe.distributions.multivariate import MvNormal

class TestMvNormal(unittest.TestCase):

    def test_sample(self):
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        dist = MvNormal(mean, cov)
        samples = dist.sample(10)
        self.assertEqual(samples.shape, (10, 2))

    def test_mean_cov(self):
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.2], [0.2, 2.0]])
        dist = MvNormal(mean, cov)
        np.testing.assert_allclose(dist.mean(), mean)
        np.testing.assert_allclose(dist.cov(), cov)







if __name__ == "__main__":
    unittest.main()