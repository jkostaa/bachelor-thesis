import numpy as np
import matplotlib.pyplot as plt

import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MAP.map_tv_minimize import MAPEstimator

class MMSEEstimatorULA:
    def __init__(self, M, sigma, lambda_, eps, ula_step_size, burn_in, thin, num_samples):
        self.ula_step_size = ula_step_size
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples
        self.map_estimator = MAPEstimator(M, sigma, lambda_, eps) # MAP instance for gradient computations

    def ula_sampling(self, y, x_init=None):
        #x = np.zeros_like(np.fft.ifft2(y).real)

        if x_init is None:
            x = np.real(np.fft.ifft2(y))
        else:
            x = np.array(x_init, dtype=float)

        energies = []
        samples_kept = []
        n_iters = int(self.burn_in + max(0, int(self.num_samples)) * max(1, int(self.thin)))
        for i in range(n_iters):
        #for i in range(self.burn_in + self.num_samples * self.thin):
        
        

            '''
            #E_before = self.map_estimator.compute_loss(x, y)
            #g_data = self.map_estimator.data_fidelity_gradient(x, y)
            #g_tv = self.map_estimator.huber_tv_subgradient(x)

            #dx, dy = self.map_estimator.finite_diff_gradient(x)   # use your chosen diff (Neumann/periodic)
            #t = np.sqrt(dx*dx + dy*dy)
            #eps_safe = max(self.map_estimator.eps, 1e-12)
            #w = np.where(t <= eps_safe, 1.0/eps_safe, 1.0/(t + 1e-12))
            #px = w * dx
            #py = w * dy
            #div = -self.map_estimator.divergence(px, py)

            
            # if i == 0 or i % 50 == 0:
            #     print(f"ITERATION {i} FOR TESTING THE TV COMPONENTS")
            #     print("dx norm, dy norm:", np.linalg.norm(dx), np.linalg.norm(dy))
            #     print("t min/median/mean/max:", t.min(), np.median(t), t.mean(), t.max())

            #     print("w min/median/mean/max:", w.min(), np.median(w), w.mean(), w.max())

            #     print("px norm, py norm:", np.linalg.norm(px), np.linalg.norm(py))

            #     print("div norm (this is g_tv):", np.linalg.norm(div))
            '''


            grad = (self.map_estimator.data_fidelity_gradient(x, y) + self.map_estimator.lambda_ * self.map_estimator.huber_tv_subgradient(x))
            
            gnorm = np.linalg.norm(grad)
            if i % 25 == 0:
                print(f"ULA iter {i:4d}, grad_norm={gnorm:.3e}, sample_mean={x.mean():.3e}, sample_std={x.std():.3e}")

            # if i == 0 or i % 25 == 0:
            #     print (f"ITERATION: {i}")
            #     print("E_before:", E_before)
            #     print("||grad||, ||g_data||, ||g_tv||:", np.linalg.norm(grad), np.linalg.norm(g_data), np.linalg.norm(g_tv))
            #     print("x min/max:", x.min(), x.max(), "mean:", x.mean())

            noise = np.random.randn(*x.shape) * np.sqrt(2.0 * self.ula_step_size)
            #drift = self.ula_step_size * grad

            x = x - grad * self.ula_step_size + noise # ULA update: x_{t+1} = x_t - step * ∇U(x_t) + sqrt(2*step)*N(0, I)*


            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("NaN/Inf detected at iteration", i)
                break

            # if i == 0 or i % 25 == 0:            
            #     print("drift_norm:", np.linalg.norm(drift), "noise_norm:", np.linalg.norm(noise),
            #     "ratio noise/drift:", np.linalg.norm(noise)/(np.linalg.norm(drift)+1e-16))

            energy = self.map_estimator.compute_loss(x, y) # energy = negative log posterior (tracked per sample)
            energies.append(energy)

            if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                samples_kept.append(np.copy(x))
            
        if len(samples_kept) == 0:
            print("Warning: no samples kept (increase num_samples or decrease burn_in). returning last state as single sample.")
            samples_kept = [np.copy(x)]

        # if len(samples_kept) == 0:
        #     mmse_estimate = x
        # else:
        #     mmse_estimate = np.mean(np.stack(samples_kept, axis=0), axis=0)


            # if i == 0 or i % 10 == 0:     
            #     plt.figure(figsize=(10,4))
            #     plt.subplot(1,3,1); plt.imshow(dx, cmap='RdBu'); plt.title('dx'); plt.colorbar()
            #     plt.subplot(1,3,2); plt.imshow(dy, cmap='RdBu'); plt.title('dy'); plt.colorbar()
            #     plt.subplot(1,3,3); plt.imshow(div, cmap='RdBu'); plt.title('g_tv (div)'); plt.colorbar()
            #     plt.show()

        return samples_kept, energies
    
    def compute_mmse_estimate(self, y, x_init=None):
        """
        Compute MMSE estimate = mean over ULA posterior samples.
        """
        samples, energies = self.ula_sampling(y)
        print(len(samples))
        try:
            arr = np.stack(samples, axis=0)   # shape (N, H, W)
        except Exception as e:
            print("Failed to stack samples:", e)
            print("Sample shapes:", [np.shape(s) for s in samples])
            # Fallback: use last sample as single-item array
            arr = np.expand_dims(np.array(samples[-1]), axis=0)

        x_mmse = np.mean(arr, axis=0)
        x_mmse = np.nan_to_num(x_mmse)
        denom = x_mmse.max() - x_mmse.min()
        if denom > 0:
            normalized_mmse = (x_mmse - x_mmse.min()) / denom
        else:
            normalized_mmse = x_mmse
        return normalized_mmse
    
    def compute_variance_map(self, y, x_init=None):
        """Compute pixelwise variance map from posterior samples."""
        samples, energies = self.ula_sampling(y)
        variance_map = np.var(samples, axis=0)
        return variance_map

        '''
    def plot_variance_map(self, variance_map, cmap="magma", vmax=None):
        """
        Visualize the posterior variance map.

        variance_map : ndarray
        cmap : str
        vmax : float or None
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(variance_map, cmap=cmap, vmax=vmax)
        plt.colorbar(label="Posterior Variance")
        plt.title("Posterior Uncertainty Map (Variance)")
        plt.axis("off")

        plt.show()
        '''