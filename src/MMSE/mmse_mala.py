import numpy as np

np.random.seed(42) # for reproducibility

class MMSEEstimatorMALA:
    def __init__(self, M, sigma, lambda_, eps, mala_step_size, burn_in, thin, num_samples):
        self.M = M
        self.sigma = sigma
        self.lambda_ = lambda_
        self.eps = eps
        self.mala_step_size = mala_step_size
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples
        self.sigma = sigma
        self.lambda_ = lambda_

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

        return div

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
    
    def negative_log_posterior(self, x, y):
        # U(x) = 0.5 / sigma^2 * ||M(x)-y||^2 + lambda*TV(x)
        data_term = 0.5 / self.sigma**2 * np.linalg.norm(self.A(x) - y)**2
        #reg_term = self.lambda_ * np.sum(np.sqrt(self.huber_tv_subgradient(x)**2 + 1e-8))
        # keep reg term simple (sum of abs of TV-subgradient)

        reg_term = self.lambda_ * self.huber_tv_2d(x) # should use self.huber_tv_2d(x) ...
        return data_term + reg_term

    def log_q(self, x_from, x_to, grad_from, step):
        # log of Gaussian proposal density: N(x_from - step*grad_from, 2*step*I)
        diff = x_to - x_from + 0.5 * step * grad_from
        return -0.25 / step * np.sum(diff**2)

    def gradient_check(self, x_test, y, i=10, j=10, eps=1e-6): # diagnostic function for data fidelity gradient check
        """
        Diagnostic finite-difference test: compares analytical and numerical gradients.
        """

        # --- 1. Analytical gradient ---
        x0 = np.real(self.A_adj(y)).astype(np.complex128)             # use your typical x_init
        eps = 1e-6
        i, j = 10, 10

        # Analytical gradient (data term only)
        grad_anal = (self.data_fidelity_gradient(x0, y))[i, j]   # complex value

        # Numerical gradient (perturb real part)
        x_p = x0.copy(); x_p[i, j] += eps
        x_m = x0.copy(); x_m[i, j] -= eps

        U_data = lambda z: 0.5 / (self.sigma**2) * np.linalg.norm(self.A(z) - y)**2
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
        grad_anal = self.huber_tv_subgradient(x)[i, j]

        # Compute numerical gradient
        eps = 1e-5
        x_plus  = x.copy(); x_plus[i, j]  += eps
        x_minus = x.copy(); x_minus[i, j] -= eps

        tv_plus  = self.huber_tv_2d(x_plus)
        tv_minus = self.huber_tv_2d(x_minus)

        grad_num = (tv_plus - tv_minus) / (2 * eps)

        print("Analytical:", grad_anal)
        print("Numerical :", grad_num)
        print("Abs err   :", abs(grad_anal - grad_num))
        print("Rel err   :", abs(grad_anal - grad_num) / (abs(grad_num) + 1e-12))

    def energy_check(self, x, y):
        data_term = 0.5 / (self.sigma**2) * np.linalg.norm(self.A(x) - y)**2
        reg_term  = self.lambda_ * self.huber_tv_2d(x)
        U = self.negative_log_posterior(x, y)   # or energy(x)

        print("data_term:", data_term)
        print("reg_term: ", reg_term)
        print("U (energy):", U)
        print("sum parts:", data_term + reg_term)
        print("abs diff:", abs(U - (data_term + reg_term)))

    # def mala_sampling(self, y, x_init=None):
    #     # x = np.zeros_like(np.fft.ifft2(y).real)

    #     if x_init is None:
    #         x = np.real(np.fft.ifft2(y)).astype(np.float64)
    #         # x = np.real(self.A_adj(y))
    #     else:
    #         x = np.array(x_init, dtype=np.float64)

    #     samples_kept = []
    #     accepted = 0
    #     total = 0

    #     # extra diagnostics
    #     energies = []
    #     accept_trace = []
    #     step_trace = []

    #     for i in range(self.burn_in + self.num_samples * self.thin):
    #         grad = (self.data_fidelity_gradient(x, y)
    #                 + self.lambda_ * self.huber_tv_subgradient(x)).astype(np.float64)
            
    #         #print(grad.dtype)

    #         noise = np.random.randn(*x.shape) * np.sqrt(2.0 * self.mala_step_size)
    #         x_proposal = x - grad * self.mala_step_size + noise

    #         # Compute acceptance probability
    #         U_x = self.negative_log_posterior(x, y)
    #         U_xp = self.negative_log_posterior(x_proposal, y)

    #         grad_proposal = (self.data_fidelity_gradient(x_proposal, y)
    #                          + self.lambda_ * self.huber_tv_subgradient(x_proposal))

    #         log_alpha = -U_xp + U_x + self.log_q(x_proposal, x, grad_proposal) - self.log_q(x, x_proposal, grad)

    #         if log_alpha > 0:
    #             alpha = 1.0
    #         else:
    #             alpha = np.exp(log_alpha)

    #         # or: alpha = min(1.0, np.exp(np.clip(log_alpha, a_min=None, a_max=0)))

    #         # Accept/reject

    #         if np.random.rand() < alpha:
    #             x = x_proposal
    #             accepted += 1
    #             accepted_flag = 1
    #         else:
    #             accepted_flag = 0
    #         total += 1

    #         energies.append(U_x)   # store current energy (before possible accept)
    #         accept_trace.append(accepted_flag)
    #         step_trace.append(self.mala_step_size)

    #         if np.any(np.isnan(x)) or np.any(np.isinf(x)):
    #             print("NaN/Inf detected at iteration", i)
    #             break

    #         # Store samples after burn-in with thinning
    #         if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
    #             samples_kept.append(np.copy(x))

    #         # optional step adaptation (very simple): target acceptance ~0.57 for MALA
    #         if (i + 1) % 200 == 0 and i < self.burn_in:
    #             recent_accept = np.mean(accept_trace[-200:])
    #             if recent_accept > 0.65:
    #                 self.mala_step_size *= 1.2
    #             elif recent_accept < 0.45:
    #                 self.mala_step_size *= 0.7
    #             # keep step in reasonable bounds
    #             self.mala_step_size = float(np.clip(self.mala_step_size, 1e-12, 1e-1))

    #     print(f"accept_rate: {accepted / total}")
    #     # print(f"accept_trace: {np.array(accept_trace)}, accept_rate: {accepted / total}, step_trace: {np.array(step_trace)}")
    #     return samples_kept, energies, step_trace, accept_trace

    def mala_sampling(self, y, x_init=None, debug=False, clip_grad=None):
        """
        Standard MALA:
          proposal: x' = x - 0.5 * eta * gradU(x) + sqrt(eta) * N(0,I)
        Returns: samples_kept, energies, step_trace, accept_trace
        """
        eta = float(self.mala_step_size)
        if x_init is None:
            x = np.real(np.fft.ifft2(y)).astype(np.float64)
        else:
            x = np.array(x_init, dtype=np.float64)

        n_iters = int(self.burn_in + max(0, int(self.num_samples)) * max(1, int(self.thin)))
        samples_kept = []
        energies = []
        accept_trace = np.zeros(n_iters, dtype=float)
        step_trace = np.full(n_iters, eta, dtype=float)

        accepted_list = []
        total_list = []

        def U(z):
            return float(self.negative_log_posterior(z, y))

        def gradU(z):
            g = (self.data_fidelity_gradient(z, y) + self.lambda_ * self.huber_tv_subgradient(z))
            return g.astype(np.float64)

        U_x = U(x)

        for i in range(n_iters):
            g_x = gradU(x) 
            # optional gradient clipping
            if clip_grad is not None:
                gnorm = np.linalg.norm(g_x)
                if gnorm > clip_grad:
                    g_x = g_x * (clip_grad / (gnorm + 1e-16))

            # proposal (standard MALA)
            z_noise = np.random.randn(*x.shape).astype(np.float64)
            x_prop = x - 0.5 * eta * g_x + np.sqrt(eta) * z_noise

            # compute target and grad at proposal
            try:
                U_prop = U(x_prop)
                g_prop = gradU(x_prop)
            except Exception:
                U_prop = np.inf
                g_prop = np.zeros_like(x)

            # asymmetric proposal correction: log q(x | x_prop) - log q(x_prop | x)
            log_q_x_given_prop = self.log_q(x_prop, x, g_prop, eta)
            log_q_prop_given_x = self.log_q(x, x_prop, g_x, eta)
            log_alpha = -U_prop + U_x + log_q_x_given_prop - log_q_prop_given_x

            # numeric safe alpha
            if log_alpha >= 0:
                alpha = 1.0
            else:
                alpha = float(np.exp(log_alpha))

            accept = (np.random.rand() < alpha)
            if accept:
                x = x_prop
                U_x = U_prop

            accept_trace[i] = 1.0 if accept else 0.0
            step_trace[i] = eta
            energies.append(U_x)

            # update accepted/total lists
            accepted_list.append(1 if accept else 0)
            total_list.append(1)

            if debug and (i % 50 == 0):
                data_term = 0.5 * np.linalg.norm(self.A(x) - y)**2 / self.sigma**2
                prior_term = self.lambda_ * self.huber_tv_2d(x)
                print(f"iter {i:4d} data={data_term:.3e} prior={prior_term:.3e} total={data_term + prior_term:.3e}")
                #print(f"MALA iter {i:4d} U={U_x:.4e} accept_rate={accept_trace[:i+1].mean():.3f} gnorm={np.linalg.norm(g_x):.3e}")

            if not np.all(np.isfinite(x)):
                print("Non-finite state at iter", i, "- stopping.")
                break

            if debug and i % 100 == 0:
                TV = self.huber_tv_2d(x) # current TV(x)
                data_term = np.linalg.norm(self.A(x) - y)**2 / (2 * self.sigma**2)  # current data term

                r = 0.1  # target prior : data ratio (10%)
                lambda_needed = (r * data_term) / (TV + 1e-12)

                print(f"TV={TV:.4f}, data={data_term:.3e}, lambda_needed={lambda_needed:.3f}")

            if debug and i % 100 == 0:
                g_data = self.data_fidelity_gradient(x, y)
                g_prior = self.huber_tv_subgradient(x)

                print("‖g_data‖ =", np.linalg.norm(g_data))
                print("‖g_prior‖ =", np.linalg.norm(g_prior))
                print("ratio g_prior/g_data =", np.linalg.norm(g_prior)/np.linalg.norm(g_data))

            # store after burn-in with thinning using class params
            if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                samples_kept.append(np.copy(x))

            # # safe adaptive step tuning during burn-in (ensure enough history)
            # if (i + 1) % 200 == 0 and i < self.burn_in:
            #     window = min(200, len(accept_trace[:i+1]))
            #     recent_accept = np.mean(accept_trace[max(0, i+1-window):i+1])
            #     if recent_accept > 0.65:
            #         eta *= 1.2
            #     elif recent_accept < 0.45:
            #         eta *= 0.7
            #     eta = float(np.clip(eta, 1e-12, 1e-1))
            #     self.mala_step_size = eta  # update stored step

            # gentle adaptive tuning during burn-in (multiplicative log-update)
            if (i + 1) % 30 == 0 and i < self.burn_in:
                adapt_window = min(30, i + 1)
                min_window = 30  # require at least this many iters to adapt
                if adapt_window >= min_window:
                    recent_accept = np.mean(accept_trace[i + 1 - adapt_window : i + 1])
                    # target in [0.4, 0.6]; use central target
                    target_accept = 0.8
                    # small gain for stability (increase => larger steps when accept>target)
                    gamma = 0.95
                    eta *= float(np.exp(gamma * (recent_accept - target_accept)))
                    eta = float(np.clip(eta, 1e-12, 1e-1))
                    self.mala_step_size = eta    

        # compute and print acceptance rate using the lists
        total = sum(total_list) if len(total_list) > 0 else 0
        accepted = sum(accepted_list) if len(accepted_list) > 0 else 0
        acc_rate = accepted / total if total > 0 else 0.0
        print(f"MALA acceptance rate: {acc_rate:.4f} ({accepted}/{total})")

        # fallback
        if len(samples_kept) == 0:
            print("Warning: no samples kept; returning last state.")
            samples_kept = [np.copy(x)]

        return samples_kept, energies, step_trace[:len(energies)], accept_trace[:len(energies)]
    
    def compute_mmse_estimate(self, samples):
        """
        Compute MMSE estimate = mean over MALA posterior samples.
        """
        #samples, energies, step_trace, accept_trace = self.mala_sampling(y)
        try:
            arr = np.stack(samples, axis=0)   # shape (N, H, W)
        except Exception as e:
            print("Failed to stack samples:", e)
            print("Sample shapes:", [np.shape(s) for s in samples])
            arr = np.expand_dims(np.array(samples[-1]), axis=0)

        x_mmse = np.mean(arr, axis=0)
        x_mmse = np.nan_to_num(x_mmse)
        denom = x_mmse.max() - x_mmse.min()
        if denom > 0:
            normalized_mmse = (x_mmse - x_mmse.min()) / denom
        else:
            normalized_mmse = x_mmse
        return normalized_mmse
    
    def test_adjoint_A(est, H, W):
        x = np.random.randn(H, W) + 1j*np.random.randn(H, W)
        y = np.random.randn(H, W) + 1j*np.random.randn(H, W)
        lhs = np.vdot(est.A(x), y)
        rhs = np.vdot(x, est.A_adj(y))
        print("A adjoint diff:", lhs - rhs)

    def test_grad_div_adjoint(est, H, W):
        x = np.random.randn(H, W)
        p = np.random.randn(H, W)
        q = np.random.randn(H, W)
        gx, gy = est.finite_diff_gradient(x)
        lhs = np.sum(gx * p + gy * q)
        rhs = -np.sum(x * est.divergence(p, q))
        print("grad-div diff:", lhs - rhs)

    def test_full_gradient(est, H, W, y):
        x = np.random.randn(H, W)
        d = np.random.randn(H, W)
        d /= np.linalg.norm(d)  # normalize direction
        eps = 1e-6
        g_analytic = est.data_fidelity_gradient(x, y) + est.lambda_ * est.huber_tv_subgradient(x)
        ana = np.sum(g_analytic * d)
        Eplus  = est.compute_loss(x + eps*d, y)
        Eminus = est.compute_loss(x - eps*d, y)
        num = (Eplus - Eminus) / (2*eps)
        print("full grad diff:", ana - num, "relative:", abs(ana-num)/max(1e-12, abs(num)))