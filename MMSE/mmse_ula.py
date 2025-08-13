import numpy as np

import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MAP.map_tv_minimize import MAPEstimator

class MMSEEstimatorULA:
    def __init__(self, M, sigma, lambda_, eps, learning_rate, max_iters, ula_step_size, burn_in, thin, num_samples):
        self.ula_step_size = ula_step_size
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples
        self.map_estimator = MAPEstimator(M, sigma, lambda_, eps, learning_rate, max_iters) # MAP instance for gradient computations

    def ula_mmse(self, y):
        x = np.zeros_like(np.fft.ifft2(y).real)
        
        energies = []
        samples_kept = []

        for i in range(self.burn_in + self.num_samples * self.thin):
            grad = (self.map_estimator.data_fidelity_gradient(x, y) + self.map_estimator.lambda_ * self.map_estimator.huber_tv_subgradient(x))
            
            noise = np.random.randn(*x.shape) * np.sqrt(2 * self.ula_step_size)
            x = x - grad * self.ula_step_size + noise # ULA update: x_{t+1} = x_t - step * ∇U(x_t) + sqrt(2*step)*N(0, I)

            energy = self.map_estimator.compute_loss(x, y) # energy = negative log posterior (tracked per sample)
            energies.append(energy)

            if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                samples_kept.append(np.copy(x))

        # if len(samples_kept) == 0:
        #     mmse_estimate = x
        # else:
        #     mmse_estimate = np.mean(np.stack(samples_kept, axis=0), axis=0)

        return samples_kept, energies