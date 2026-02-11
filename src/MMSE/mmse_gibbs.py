import numpy as np
from scipy.stats import invgamma
from tqdm import trange

import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MAP.map_tv_minimize import MAPEstimator

np.random.seed(42) # for reproducibility


class MMSEEstimatorGibbs:
    def __init__(self, mask, eps, sigma=0.01, lambda_=0.1, step_size=1e-3,
                 sigma_prior=1e-2, x_mala_steps=5, num_iters=50):
        """
        Gibbs sampler for MRI reconstruction:
        Alternates between sampling x | sigma² and sigma² | x.

        Args:
            mask: undersampling mask (numpy array)
            sigma: initial noise standard deviation
            lambda_: TV regularization weight
            step_size: Langevin/MALA step size
            sigma_prior: prior scale for sigma²
            x_mala_steps: number of inner MALA steps per iteration
            num_iters: total Gibbs iterations
        """
        self.mask = mask
        self.eps = eps
        self.sigma = sigma
        self.lambda_ = lambda_
        self.step_size = step_size
        self.sigma_prior = sigma_prior
        self.x_mala_steps = x_mala_steps
        self.num_iters = num_iters
        self.samples = []
        self.map_estimator = MAPEstimator(self.mask, self.sigma, self.lambda_, self.eps)

    def negative_log_posterior(self, x, y):
        # U(x) = 0.5 / sigma^2 * ||M(x)-y||^2 + lambda*TV(x)
        data_term = 0.5 / self.sigma**2 * np.linalg.norm(self.map_estimator.A(x) - y)**2
        #reg_term = self.lambda_ * np.sum(np.sqrt(self.map_estimator.huber_tv_subgradient(x)**2 + 1e-8))
        # keep reg term simple (sum of abs of TV-subgradient)

        reg_term = self.lambda_ * np.sum(np.abs(self.map_estimator.huber_tv_subgradient(x)))
        return data_term + reg_term

    def mala_single_step(self, map_est, x, y):
        """Perform a single MALA step on x | y, sigma²"""
        grad = map_est.data_fidelity_gradient(x, y) + map_est.huber_tv_subgradient(x)
        noise = np.random.randn(*x.shape) * np.sqrt(2 * self.step_size)
        x_prop = x - self.step_size * grad + noise

        # Compute acceptance probability
        log_post_x = -self.negative_log_posterior(x, y)
        log_post_x_prop = -self.negative_log_posterior(x_prop, y)
        log_alpha = log_post_x_prop - log_post_x

        if np.isnan(log_alpha):
            log_alpha = -np.inf
        if log_alpha > 0 or np.log(np.random.rand()) < log_alpha:
            return x_prop, True
        return x, False

    def sample(self, y):
        """Run the Gibbs sampling loop"""

        map_est = MAPEstimator(self.mask, self.sigma, self.lambda_, 1e-6)
        x = np.real(np.fft.ifft2(y))  # initialize in image domain

        acc_x = 0
        for it in trange(self.num_iters, desc="Gibbs Sampling"):
            # --- Sample x via MALA-within-Gibbs ---
            for _ in range(self.x_mala_steps):
                x, accepted = self.mala_single_step(map_est, x, y)
                acc_x += int(accepted)

            # --- Sample sigma² via inverse-gamma ---
            residual = map_est.A(x) - y
            shape = np.prod(y.shape) / 2 + 1
            scale = np.sum(np.abs(residual) ** 2) / 2 + self.sigma_prior
            self.sigma = np.sqrt(1 / np.random.gamma(shape, 1 / scale))

            # --- Store sample ---
            self.samples.append(x.copy())

        print(f"Mean acceptance rate: {acc_x / (self.num_iters * self.x_mala_steps):.3f}")
        return np.array(self.samples)

    def compute_mmse_estimate(self, y):
        """Compute MMSE estimate (posterior mean of samples)"""
        if len(self.samples) == 0:
            raise RuntimeError("Run sample(y) before computing MMSE estimate.")
        x_mmse = np.mean(self.samples, axis=0)
        # Normalize for visualization
        x_mmse = np.nan_to_num(x_mmse)
        x_mmse = (x_mmse - x_mmse.min()) / (x_mmse.max() - x_mmse.min() + 1e-12)
        return x_mmse