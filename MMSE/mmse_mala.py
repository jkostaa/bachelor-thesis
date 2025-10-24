import numpy as np

import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MAP.map_tv_minimize import MAPEstimator

np.random.seed(42) # for reproducibility

class MMSEEstimatorMALA:
    def __init__(self, M, sigma, lambda_, eps, mala_step_size, burn_in, thin, num_samples):
        self.mala_step_size = mala_step_size
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples
        self.map_estimator = MAPEstimator(M, sigma, lambda_, eps)  # MAP instance for gradient computations
        self.sigma = sigma
        self.lambda_ = lambda_

    def negative_log_posterior(self, x, y):
        # U(x) = 0.5 / sigma^2 * ||M(x)-y||^2 + lambda*TV(x)
        data_term = 0.5 / self.sigma**2 * np.linalg.norm(self.map_estimator.A(x) - y)**2
        #reg_term = self.lambda_ * np.sum(np.sqrt(self.map_estimator.huber_tv_subgradient(x)**2 + 1e-8))
        # keep reg term simple (sum of abs of TV-subgradient)

        reg_term = self.lambda_ * np.sum(np.abs(self.map_estimator.huber_tv_subgradient(x)))
        return data_term + reg_term

    def log_q(self, x_from, x_to, grad_from):
        # log of Gaussian proposal density: N(x_from - step*grad_from, 2*step*I)
        diff = x_to - x_from + self.mala_step_size * grad_from
        return -0.25 / self.mala_step_size * np.sum(diff**2)

    def mala_sampling(self, y, x_init=None):
        # x = np.zeros_like(np.fft.ifft2(y).real)

        if x_init is None:
            x = np.real(np.fft.ifft2(y))
        else:
            x = np.array(x_init, dtype=float)

        samples_kept = []

        for i in range(self.burn_in + self.num_samples * self.thin):
            grad = (self.map_estimator.data_fidelity_gradient(x, y)
                    + self.lambda_ * self.map_estimator.huber_tv_subgradient(x))

            #print(grad)

            noise = np.random.randn(*x.shape) * np.sqrt(2.0 * self.mala_step_size)
            x_proposal = x - grad * self.mala_step_size + noise

            # Compute acceptance probability
            U_x = self.negative_log_posterior(x, y)
            U_xp = self.negative_log_posterior(x_proposal, y)

            grad_proposal = (self.map_estimator.data_fidelity_gradient(x_proposal, y)
                             + self.lambda_ * self.map_estimator.huber_tv_subgradient(x_proposal))

            log_alpha = -U_xp + U_x + self.log_q(x_proposal, x, grad_proposal) - self.log_q(x, x_proposal, grad)

            if log_alpha > 0:
                alpha = 1.0
            else:
                alpha = np.exp(log_alpha)

            # or: alpha = min(1.0, np.exp(np.clip(log_alpha, a_min=None, a_max=0)))

            # Accept/reject
            if np.random.rand() < alpha:
                x = x_proposal

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("NaN/Inf detected at iteration", i)
                break

            # Store samples after burn-in with thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                samples_kept.append(np.copy(x))

        return samples_kept
    
    def compute_mmse_estimate(self, y, x_init=None):
        """
        Compute MMSE estimate = mean over MALA posterior samples.
        """
        samples = self.mala_sampling(y)
        x_mmse = np.mean(samples, axis=0)
        x_mmse = np.nan_to_num(x_mmse)  # replaces NaNs and Infs with 0
        denom = x_mmse.max() - x_mmse.min()
        if denom > 0:
            normalized_mmse = (x_mmse - x_mmse.min()) / denom
        else:
            normalized_mmse = x_mmse  # fallback if flat
        return normalized_mmse