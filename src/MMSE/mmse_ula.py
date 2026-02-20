import numpy as np

class MMSEEstimatorULA:
    def __init__(self, M, sigma, lambda_, eps, ula_step_size, burn_in, thin, num_samples):
        self.M = M
        self.sigma = sigma
        self.lambda_ = lambda_
        self.eps = eps
        self.ula_step_size = ula_step_size
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples

    def A(self, x):
        """
        Defined as: A = M * F
        Compute the composition of 2D discrete Fourier Transform + apply mask
        Returns a (complex) ndarray type
        """
        return self.M * np.fft.fft2(x, norm='ortho') 

    def A_adj(self, k_residual):
        """
        Compute the adjoint of A
        """
        return np.fft.ifft2(self.M * k_residual, norm='ortho')  

    # Gradients and TV

    def data_fidelity_gradient(self, x, y):
        """
        Gradient of data fidelity term:
        k_residual: Ax - y <-> M * F(x) - y ... error between predicted & measured k-space data, has dimensions of k-space(=residual)
        U_data(x) = 0.5 / sigma**2 * || A(x) - y ||^2
        grad = A^*(A x - y) / sigma^2
        Returns a complex-valued array (same shape as x)
        """

        residual = self.A(x) - y
        grad = self.A_adj(residual)  / self.sigma**2 
        return np.real(grad) #(grad.real) 

    def finite_diff_gradient(self, x):  # we are assuming that x is a 2D image
        """
        Computes the forward finite differences
        Input: 2D array
        Output: 2 arrays
        Returns the horizontal and vertical gradient
        """

        gx = np.zeros_like(x)
        gy = np.zeros_like(x)

        gx[:, :-1] = x[:, 1:] - x[:, :-1]   # horizontal (right - center)
        gy[:-1, :] = x[1:, :] - x[:-1, :]   # vertical (bottom - center)

        return gx, gy

 
    def huber_penalty_function_grad(self, dx, dy):  # =huber weights
        """
        Computes the derivative of the Huber penalty function w.r.t. dx, dy.
        """
        t = np.sqrt(dx**2 + dy**2)
        w = np.where(t <= self.eps, 1.0 / self.eps, 1.0 / (t + 1e-12))
        grad_dx = w * dx
        grad_dy = w * dy
        return grad_dx, grad_dy
    
    def divergence(self, px, py):
        """
        Computes the divergence of a vector field
        Input: 2 arrays (vector field)
        Output: 2D array (scalar)
        """

        div = np.zeros_like(px)

        # adjoint of horizontal forward diff
        div[:, 0]      -= px[:, 0]
        div[:, 1:-1]  += px[:, :-2] - px[:, 1:-1]
        div[:, -1]    += px[:, -2]

        # adjoint of vertical forward diff
        div[0, :]     -= py[0, :]
        div[1:-1, :]  += py[:-2, :] - py[1:-1, :]
        div[-1, :]    += py[-2, :]

        return -div

    def huber_tv_2d(self, x):
        """
        A function that computes the Huber total variation of a 2D array (resembling a picture made up of n x m pixels)

        Parameters:
        x: ndarray of shape (n,m) - the input image -> approximated x
        eps: float, threshold value (threshold between quadratic and linear region)
        dx: horizontal differences (n, m-1)
        dy: vertical differences (n-1, m)
        """

        dx, dy = self.finite_diff_gradient(x)

        t = np.sqrt(dx**2 + dy**2)
        quad = (t**2) / (2 * self.eps)
        lin = np.abs(t) - (self.eps / 2) # np.abs(t) or just t
        tv = np.where(t <= self.eps, quad, lin)

        return tv.sum()


    def huber_tv_subgradient(self, x):
        """
        Computes the subgradient of TV
        Return: subgradient of TV
        There is a divergence used at the end -> divergence maps back to scalar - same shape as x (image)
        """
        dx, dy = self.finite_diff_gradient(x)
        grad_dx, grad_dy = self.huber_penalty_function_grad(dx, dy)

        return -self.divergence(grad_dx, grad_dy)

    def compute_loss(self, x, y): # check maybe if data_term is correct (2* sigma or just sigma)
        '''
        Function to compute the loss (used later in the subgradient descent function
        to plot the loss over iterations)
        '''
        data_term = np.linalg.norm(self.A(x) - y) ** 2 / (2 * self.sigma**2)
        tv_term = self.lambda_ * self.huber_tv_2d(x)
        return data_term + tv_term

    def ula_sampling(self, y, x_init=None):
        #x = np.zeros_like(np.fft.ifft2(y).real)

        if x_init is None:
            x = np.real(np.fft.ifft2(y, norm='ortho')).astype(np.float64)
            # x = np.real(self.A_adj(y))
        else:
            x = np.array(x_init, dtype=np.float64)

        energies = []
        samples_kept = []
        n_iters = int(self.burn_in + max(0, int(self.num_samples)) * max(1, int(self.thin)))
        
        for i in range(n_iters):
            
            # compute gradient of U = data_term + lambda * TV
            grad = (self.data_fidelity_gradient(x, y) + self.lambda_ * self.huber_tv_subgradient(x))
            
            gnorm = np.linalg.norm(grad)
            # if gnorm > 1e2: # 1e2 = clip
            #     grad = grad * (1e2 / (gnorm + 1e-16))
            #     gnorm = 1e2
            if i % 50 == 0:
                print(f"ULA iter {i:4d}, min={grad.min():.3e}, max={grad.max():.3e} grad_norm={gnorm:.3e}, sample_mean={x.mean():.3e}, sample_std={x.std():.3e}")


            noise = np.random.randn(*x.shape) * np.sqrt(2.0 * self.ula_step_size)
            #drift = self.ula_step_size * grad

            # x_prev = x.copy()
            x = x - grad * self.ula_step_size + noise  # ULA update: x_{t+1} = x_t - step * ∇U(x_t) + sqrt(2*step)*N(0, I)*
            

            # diagnostic

            # try:
            #     U_prev = float(self.compute_loss(x_prev, y))
            #     U_now  = float(self.compute_loss(x, y))
            # except Exception as e:
            #     print("Energy computation failed:", e)
            #     U_prev, U_now = np.inf, np.inf

            # if i % 10 == 0:
            #     print(f"iter {i:4d}  U_prev={U_prev:.4e}  U_now={U_now:.4e}  dU={U_now-U_prev:.4e}  gnorm={np.linalg.norm(grad):.3e} drift={np.linalg.norm(self.ula_step_size * grad):.3e} noise={np.linalg.norm(noise):.3e}")

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("NaN/Inf detected at iteration", i)
                break

            # if i == 0 or i % 25 == 0:            
            #     print("drift_norm:", np.linalg.norm(drift), "noise_norm:", np.linalg.norm(noise),
            #     "ratio noise/drift:", np.linalg.norm(noise)/(np.linalg.norm(drift)+1e-16))

            energy = float(self.compute_loss(x, y)) # energy = negative log posterior (tracked per sample)
            energies.append(energy)

            if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                samples_kept.append(np.copy(x))
            
        if len(samples_kept) == 0:
            print("Warning: no samples kept (increase num_samples or decrease burn_in). returning last state as single sample.")
            samples_kept = [np.copy(x)]

        return samples_kept, energies
    
    def compute_mmse_estimate(self, samples):
        """
        Compute MMSE estimate = mean over ULA posterior samples.
        """
        
        try:
            arr = np.stack(samples, axis=0)   # shape (N, H, W)
        except Exception as e:
            print("Failed to stack samples:", e)
            print("Sample shapes:", [np.shape(s) for s in samples])
            # Fallback: use last sample as single-item array
            arr = np.expand_dims(np.array(samples[-1]), axis=0)

        return np.mean(arr, axis=0) 
    
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


    # def autocorr(self, x): # diagnsotic
    #     # If autocorrelation stays >0.9 → not moving → step too small
    #     # f oscillatory → step too large
    #     # If decays smoothly → good
    #     x = x - x.mean()
    #     corr = np.correlate(x, x, mode='full')
    #     corr = corr[corr.size//2:]
    #     return corr / corr[0]
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