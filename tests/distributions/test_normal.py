import unittest
import numpy as np
from probpipe.core.multivariate import Normal

class TestNormal(unittest.TestCase):

    def setUp(self):
        self.mu = 1.5
        self.sigma = 2.5
        # fix RNG for reproducibility
        self.dist = Normal(mu=self.mu,
                           sigma=self.sigma,
                           rng=np.random.default_rng(123))

    def test_sample_shape(self):
        # single sample → shape (1,1)
        x1 = self.dist.sample(1)
        self.assertIsInstance(x1, np.ndarray)
        self.assertEqual(x1.shape, (1, 1))

        # multiple samples → shape (n,1)
        x10 = self.dist.sample(10)
        self.assertEqual(x10.shape, (10, 1))

    def test_mean_cov(self):
        m = self.dist.mean()
        C = self.dist.cov()

        # mean should be (1,), cov should be (1,1)
        self.assertIsInstance(m, np.ndarray)
        self.assertIsInstance(C, np.ndarray)
        self.assertEqual(m.shape, (1,))
        self.assertEqual(C.shape, (1,1))

        # values should match the parameters
        np.testing.assert_allclose(m[0], self.mu, rtol=1e-7)
        np.testing.assert_allclose(C[0,0], self.sigma**2, rtol=1e-7)

    def test_density_log_density_cdf_shapes(self):
        # Prepare inputs of various shapes
        scalar = 0.0
        arr1d = np.linspace(-1, 1, 5)         # shape (5,)
        arr2d = arr1d.reshape(5,1)            # shape (5,1)

        for fn in (self.dist.density, self.dist.log_density, self.dist.cdf):
            # scalar input → (1,1)
            out0 = fn(scalar)
            self.assertEqual(out0.shape, (1,1))

            # 1-d input → (5,1)
            out1 = fn(arr1d)
            self.assertEqual(out1.shape, (5,1))

            # 2-d input → (5,1)
            out2 = fn(arr2d)
            self.assertEqual(out2.shape, (5,1))

    def test_log_density_consistency(self):
        x = np.array([-2.0, 0.0, 2.0])
        pdf = self.dist.density(x).ravel()
        logpdf = self.dist.log_density(x).ravel()
        np.testing.assert_allclose(logpdf, np.log(pdf), rtol=1e-7)

    def test_cdf_value_range(self):
        x = np.array([-3.0, 0.0, 3.0])
        cdf = self.dist.cdf(x)
        # cdf should be in (0,1)
        self.assertTrue(np.all(cdf > 0.0))
        self.assertTrue(np.all(cdf < 1.0))

    def test_inv_cdf_shape_and_errors(self):
        # scalar → (1,1)
        u0 = self.dist.inv_cdf(0.5)
        self.assertEqual(u0.shape, (1,1))

        # numpy scalar → (1,1)
        u1 = self.dist.inv_cdf(np.array(0.5))
        self.assertEqual(u1.shape, (1,1))

        # 1-d array → (n,1)
        u_1d = np.linspace(0.1, 0.9, 7)
        out1 = self.dist.inv_cdf(u_1d)
        self.assertEqual(out1.shape, (7,1))

        # 2-d array (n,1) → (n,1)
        u_2d = u_1d.reshape(7,1)
        out2 = self.dist.inv_cdf(u_2d)
        self.assertEqual(out2.shape, (7,1))

        # invalid shape should raise
        with self.assertRaises(ValueError):
            self.dist.inv_cdf(np.zeros((3,2)))

if __name__ == "__main__":
    unittest.main()