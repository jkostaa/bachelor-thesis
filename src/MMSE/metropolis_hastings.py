import numpy as np
import matplotlib.pyplot as plt

import os
import sys


project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MAP.map_tv_minimize import MAPEstimator

class MHEstimator:
    def __init__(self, M, sigma, lambda_, eps, proposal_std, burn_in, thin, num_samples):
        """
        Metropolis–Hastings sampler.

        ----------
        proposal_std : int
            the standard deviation of the Gaussian noise you add to your current sample to propose a new one.
            Controls how big each jump in the Markov chain is.
        energy_fn : callable
            Function U(x) returning the negative log-probability (up to constant).
        proposal_fn : callable
            Function proposing a new sample given current x: x' = proposal_fn(x).
        log_q_fn : callable
            Function returning log q(x'|x) for proposal density.
        num_samples : int
            Number of samples to draw (after burn-in and thinning).
        burn_in : int
            Number of initial samples to discard.
        thin : int
            Keep one every `thin` samples.
        """
        self.proposal_std = proposal_std # 1e-2 - 1e-3
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.thin = thin
        self.map_estimator = MAPEstimator(M, sigma, lambda_, eps) # MAP instance for gradient computation etc.

    # Generate proposals/ Random-walk MH

    def mh_sampling(self, y):
        """
        Run the MH chain starting from x0.
        Returns list of accepted samples and acceptance rate.
        """
        x = np.zeros_like(np.fft.ifft2(y).real)
        #x = self.map_estimator.subgradient_descent(y)
        samples = []
        accepted = 0
        total_iters = self.burn_in + self.num_samples * self.thin

        for i in range(total_iters):
            # Propose new sample
            x_proposal = x + np.random.randn(*x.shape) * self.proposal_std ################

            # Compute energies = negative log posterior (tracked per sample)
            U = self.map_estimator.compute_loss(x, y) / x.size
            U_proposal = self.map_estimator.compute_loss(x_proposal, y) / x.size

            # Acceptance probability
            #alpha = min(1.0, np.exp(U - U_proposal))  # symmetric proposal → q cancels
            log_alpha = U - U_proposal
            if log_alpha >= 0:
                alpha = 1.0
            else:
                alpha = np.exp(log_alpha)

            # Accept/reject step
            if np.random.rand() < alpha:
                x = x_proposal
                accepted += 1

            # Save sample if past burn-in and respecting thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                samples.append(np.copy(x))

        # Monitor the acceptance rate:
        # Too high (~>70%) → increase proposal_std
        # Too low (~<20%) → decrease proposal_std
        # Goal: moderate acceptance (roughly 20–50% for large images)

        acceptance_rate = accepted / total_iters
        print("Overall acceptance rate:", acceptance_rate)

        return np.array(samples), acceptance_rate
    
    def compute_mmse_estimate_hm(self, y, x_init=None):
        """
        Compute MMSE estimate = mean over MH posterior samples.
        """
        samples, acc_rate = self.mh_sampling(y)
        x_mmse = np.mean(samples, axis=0)
        normalized_mmse = (x_mmse - x_mmse.min()) / (x_mmse.max() - x_mmse.min())
        return normalized_mmse
    
    def compute_variance_map_hm(self, y, x_init=None):
        """Compute pixelwise variance map from posterior samples."""
        samples, acc_rate = self.mh_sampling(y)
        variance_map = np.var(samples, axis=0)
        return variance_map