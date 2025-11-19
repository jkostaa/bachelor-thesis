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
        self.map_estimator = MAPEstimator(M, sigma, lambda_, eps, learning_rate=0.01, max_iters=300)  # MAP instance for gradient computations
        self.sigma = sigma
        self.lambda_ = lambda_

    def negative_log_posterior(self, x, y):
        # U(x) = 0.5 / sigma^2 * ||M(x)-y||^2 + lambda*TV(x)
        data_term = 0.5 / self.sigma**2 * np.linalg.norm(self.map_estimator.A(x) - y)**2
        #reg_term = self.lambda_ * np.sum(np.sqrt(self.map_estimator.huber_tv_subgradient(x)**2 + 1e-8))
        # keep reg term simple (sum of abs of TV-subgradient)

        reg_term = self.lambda_ * self.map_estimator.huber_tv_2d(x) # should use self.map_estimator.huber_tv_2d(x) ...
        return data_term + reg_term

    def log_q(self, x_from, x_to, grad_from):
        # log of Gaussian proposal density: N(x_from - step*grad_from, 2*step*I)
        diff = x_to - x_from + self.mala_step_size * grad_from
        return -0.25 / self.mala_step_size * np.sum(diff**2)

    def gradient_check(self, x_test, y, i=10, j=10, eps=1e-6): # diagnostic function for data fidelity gradient check
        """
        Diagnostic finite-difference test: compares analytical and numerical gradients.
        """

        # --- 1. Analytical gradient ---
        x0 = np.real(self.map_estimator.A_adj(y)).astype(np.complex128)             # use your typical x_init
        eps = 1e-6
        i, j = 10, 10

        # Analytical gradient (data term only)
        grad_anal = (self.map_estimator.data_fidelity_gradient(x0, y))[i, j]   # complex value

        # Numerical gradient (perturb real part)
        x_p = x0.copy(); x_p[i, j] += eps
        x_m = x0.copy(); x_m[i, j] -= eps

        U_data = lambda z: 0.5 / (self.sigma**2) * np.linalg.norm(self.map_estimator.A(z) - y)**2
        grad_num = (U_data(x_p) - U_data(x_m)) / (2 * eps)   # scalar (real)

        print("grad_anal (complex)   :", grad_anal)
        print("Re(grad_anal)         :", np.real(grad_anal))
        print("Im(grad_anal)         :", np.imag(grad_anal))
        print("grad_num (finite diff):", grad_num)

    def check_tv_grad(self): # diagnostic function for huber TV gradient check
        np.random.seed(0)

        # Small random test image
        x = np.random.randn(8, 8)

        # Pick a pixel index
        i, j = 3, 4

        # Compute analytic gradient
        grad_anal = self.map_estimator.huber_tv_subgradient(x)[i, j]

        # Compute numerical gradient
        eps = 1e-5
        x_plus  = x.copy(); x_plus[i, j]  += eps
        x_minus = x.copy(); x_minus[i, j] -= eps

        tv_plus  = self.map_estimator.huber_tv_2d(x_plus)
        tv_minus = self.map_estimator.huber_tv_2d(x_minus)

        grad_num = (tv_plus - tv_minus) / (2 * eps)

        print("Analytical:", grad_anal)
        print("Numerical :", grad_num)
        print("Abs err   :", abs(grad_anal - grad_num))
        print("Rel err   :", abs(grad_anal - grad_num) / (abs(grad_num) + 1e-12))

    def energy_check(self, x, y):
        data_term = 0.5 / (self.sigma**2) * np.linalg.norm(self.map_estimator.A(x) - y)**2
        reg_term  = self.lambda_ * self.map_estimator.huber_tv_2d(x)
        U = self.negative_log_posterior(x, y)   # or energy(x)

        print("data_term:", data_term)
        print("reg_term: ", reg_term)
        print("U (energy):", U)
        print("sum parts:", data_term + reg_term)
        print("abs diff:", abs(U - (data_term + reg_term)))

    def mala_sampling(self, y, x_init=None):
        # x = np.zeros_like(np.fft.ifft2(y).real)

        if x_init is None:
            # x = np.real(np.fft.ifft2(y))
            x = np.real(self.map_estimator.A_adj(y))
        else:
            x = np.array(x_init, dtype=float)

        samples_kept = []

        for i in range(self.burn_in + self.num_samples * self.thin):
            grad = (self.map_estimator.data_fidelity_gradient(x, y)
                    + self.lambda_ * self.map_estimator.huber_tv_subgradient(x))
            
            #print(grad.dtype)

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
            accepted = 0
            total = 0
            if np.random.rand() < alpha:
                x = x_proposal
                accepted += 1
            total += 1

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("NaN/Inf detected at iteration", i)
                break

            # Store samples after burn-in with thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                samples_kept.append(np.copy(x))
        print("acceptance rate:", accepted / total)
        return samples_kept
    
    def compute_mmse_estimate(self, y, x_init=None):
        """
        Compute MMSE estimate = mean over MALA posterior samples.
        """
        samples = self.mala_sampling(y)
        x_mmse = np.mean(samples, axis=0)
        # x_mmse = np.nan_to_num(x_mmse)  # replaces NaNs and Infs with 0
        # denom = x_mmse.max() - x_mmse.min()
        # if denom > 0:
        #     normalized_mmse = (x_mmse - x_mmse.min()) / denom
        # else:
        #     normalized_mmse = x_mmse  # fallback if flat
        return x_mmse