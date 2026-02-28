import numpy as np

np.random.seed(42) # for reproducibility

class MMSEEstimatorMALA:
    def __init__(self, M, sigma, lambda_, eps, L_init, burn_in, thin, num_samples, beta=1.0):
        """
        Parameters
        ----------
        L_init : float
            Initial inverse step size.  The step size is eta = 1/L, so a
            larger L means smaller, more cautious steps.
        beta   : float in (0, 1]
            Temperature / noise scale.  beta < 1 tempers the posterior,
            reducing the effective noise magnitude and making the chain mix
            more easily in high-dimensional problems.  beta = 1 (default)
            recovers the exact, un-tempered posterior.
        """
        self.M = M
        self.sigma = sigma
        self.lambda_ = lambda_
        self.eps = eps
        self.L_init = float(L_init)
        self.burn_in = burn_in
        self.thin = thin
        self.num_samples = num_samples
        self.beta = float(beta)

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
    
    def negative_log_posterior(self, x, y):
        # U(x) = 0.5 / sigma^2 * ||M(x)-y||^2 + lambda*TV(x)
        data_term = 0.5 / self.sigma**2 * np.linalg.norm(self.A(x) - y)**2
        #reg_term = self.lambda_ * np.sum(np.sqrt(self.huber_tv_subgradient(x)**2 + 1e-8))
        # keep reg term simple (sum of abs of TV-subgradient)

        reg_term = self.lambda_ * self.huber_tv_2d(x) # should use self.huber_tv_2d(x) ...
        return data_term + reg_term

    def log_q(self, x_from, x_to, grad_from, L):
        """
        Log of the Langevin proposal density (un-normalised constant dropped).

        Proposal: x_to ~ N( x_from - g/L,  (2/L)*I )
          => mean  = x_from - grad_from / L
          => var   = 2/L  (per coordinate)
          => log q = -L/4 * ||x_to - x_from + grad_from/L||^2

        This matches the reference implementation's coefficient L/4
        and is consistent with the proposal in mala_sampling below.
        """
        diff = x_to - x_from + grad_from / L
        return -L / 4.0 * np.sum(diff**2)

    # def gradient_check(self, x_test, y, i=10, j=10, eps=1e-6): # diagnostic function for data fidelity gradient check
    #     """
    #     Diagnostic finite-difference test: compares analytical and numerical gradients.
    #     """

    #     # --- 1. Analytical gradient ---
    #     x0 = np.real(self.A_adj(y)).astype(np.complex128)             # use your typical x_init
    #     eps = 1e-6
    #     i, j = 10, 10

    #     # Analytical gradient (data term only)
    #     grad_anal = (self.data_fidelity_gradient(x0, y))[i, j]   # complex value

    #     # Numerical gradient (perturb real part)
    #     x_p = x0.copy(); x_p[i, j] += eps
    #     x_m = x0.copy(); x_m[i, j] -= eps

    #     U_data = lambda z: 0.5 / (self.sigma**2) * np.linalg.norm(self.A(z) - y)**2
    #     grad_num = (U_data(x_p) - U_data(x_m)) / (2 * eps)   # scalar (real)

    #     print("grad_anal (complex)   :", grad_anal)
    #     print("Re(grad_anal)         :", np.real(grad_anal))
    #     print("Im(grad_anal)         :", np.imag(grad_anal))
    #     print("grad_num (finite diff):", grad_num)

    # def check_tv_grad(self): # diagnostic function for huber TV gradient check
    #     np.random.seed(0)

    #     # Small random test image
    #     x = np.random.randn(8, 8)

    #     # Pick a pixel index
    #     i, j = 3, 4

    #     # Compute analytic gradient
    #     grad_anal = self.huber_tv_subgradient(x)[i, j]

    #     # Compute numerical gradient
    #     eps = 1e-5
    #     x_plus  = x.copy(); x_plus[i, j]  += eps
    #     x_minus = x.copy(); x_minus[i, j] -= eps

    #     tv_plus  = self.huber_tv_2d(x_plus)
    #     tv_minus = self.huber_tv_2d(x_minus)

    #     grad_num = (tv_plus - tv_minus) / (2 * eps)

    #     print("Analytical:", grad_anal)
    #     print("Numerical :", grad_num)
    #     print("Abs err   :", abs(grad_anal - grad_num))
    #     print("Rel err   :", abs(grad_anal - grad_num) / (abs(grad_num) + 1e-12))

    # def energy_check(self, x, y):
    #     data_term = 0.5 / (self.sigma**2) * np.linalg.norm(self.A(x) - y)**2
    #     reg_term  = self.lambda_ * self.huber_tv_2d(x)
    #     U = self.negative_log_posterior(x, y)   # or energy(x)

    #     print("data_term:", data_term)
    #     print("reg_term: ", reg_term)
    #     print("U (energy):", U)
    #     print("sum parts:", data_term + reg_term)
    #     print("abs diff:", abs(U - (data_term + reg_term)))

    def _mala_step(self, x, U_x, g_x, L, y):
        """
        Perform one MALA proposal + MH accept/reject.

        Proposal (matches reference parameterisation):
            x' = x - g/L + sqrt(2/L) * beta * xi,   xi ~ N(0, I)

        The noise variance is (2/L)*beta^2.  With beta=1 this is the
        standard Langevin discretisation.  beta<1 tempers the posterior.

        Returns
        -------
        x_new, U_new, g_new, alpha, accepted
        """
        xi = np.random.randn(*x.shape).astype(np.float64) # noise
        x_prop = x - g_x / L + np.sqrt(2.0 / L) * self.beta * xi # proposal

        try:
            U_prop = float(self.negative_log_posterior(x_prop, y))
            g_prop = (self.data_fidelity_gradient(x_prop, y)
                      + self.lambda_ * self.huber_tv_subgradient(x_prop)).astype(np.float64)
        except Exception:
            return x, U_x, g_x, 0.0, False

        # MH correction: log alpha = -U(x') + U(x) + log q(x|x') - log q(x'|x)
        # log q uses -L/4 * ||diff||^2  (see log_q docstring)
        log_q_x_given_prop = self.log_q(x_prop, x, g_prop, L) # q(x | x')
        log_q_prop_given_x = self.log_q(x,      x_prop, g_x, L) # q(x' | x)
        log_alpha = -U_prop + U_x + log_q_x_given_prop - log_q_prop_given_x

        alpha = float(np.clip(np.exp(log_alpha), 0.0, 1.0))
        accepted = (np.random.rand() < alpha)

        if accepted:
            return x_prop, U_prop, g_prop, alpha, True
        else:
            return x, U_x, g_x, alpha, False

    def mala_sampling(self, y, x_init=None, debug=False, freeze_L_after_burnin=True):
        """
        Two-phase MALA.

        Phase 1 — Burn-in (n = self.burn_in iterations):
            Run the chain with per-iteration step-size adaptation.
            L is multiplied by 1.1 when alpha < 0.574 (step too large)
            and divided by 1.1 when alpha >= 0.574 (step too small).
            No samples are collected.

        Phase 2 — Averaging (a = self.num_samples * self.thin iterations):
            Collect every self.thin-th sample to form the MMSE estimate.
            -> freeze_L_after_burnin=True (default): L is held fixed at the
            value found during burn-in.  This prevents the wild step-size
            oscillations caused by the reference's aggressive x2/div1.5
            actors, which were tuned for a learned neural prior, not Huber-TV.
            -> freeze_L_after_burnin=False: the same gentle 1.1 factor used in
            burn-in continues, which is a safe adaptive compromise.

        The proposal is:
            x' = x - g/L + sqrt(2/L) * beta * xi,   xi ~ N(0,I)

        which uses the inverse-step-size L instead of eta=1/L.
        self.beta < 1 tempers the posterior (reduces effective noise),
        helping the chain mix in high-dimensional problems.

        Returns
        -------
        samples_kept : list of np.ndarray  (length self.num_samples)
        energies     : list of float
        L_trace      : np.ndarray  (step size history, stored as 1/L for readability)
        accept_trace : np.ndarray
        """
        # ---------- initialisation ----------
        L = self.L_init

        if x_init is None:
            # adjoint-based zero-fill is more principled than bare ifft2
            x = np.real(self.A_adj(y)).astype(np.float64)
        else:
            x = np.array(x_init, dtype=np.float64)

        def gradU(z):
            return (self.data_fidelity_gradient(z, y)
                    + self.lambda_ * self.huber_tv_subgradient(z)).astype(np.float64)

        U_x = float(self.negative_log_posterior(x, y))
        g_x = gradU(x)

        if debug:
            gnorm = np.linalg.norm(g_x)
            print(f"[MALA init]  U={U_x:.4e}  ||g||={gnorm:.4e}")
            print(f"             L={L:.4e}  drift=||g||/L={gnorm/L:.4e}")
            print(f"             noise=sqrt(2/L)*beta*sqrt(d)="
                  f"{np.sqrt(2.0/L)*self.beta*np.sqrt(x.size):.4e}")

        # ---------- Phase 1: burn-in with per-step adaptation ----------
        n_burn = int(self.burn_in)
        burn_accepts = 0

        for i in range(n_burn):
            x, U_x, g_x, alpha, accepted = self._mala_step(x, U_x, g_x, L, y)

            # per-iteration multiplicative adaptation (same logic as reference)
            if alpha < 0.574:
                L *= 1.1   # acceptance too low  -> step too large -> increase L
            else:
                L /= 1.1   # acceptance too high -> step too small -> decrease L
            L = float(np.clip(L, 1e-3, 1e15))

            burn_accepts += int(accepted)

            if not np.all(np.isfinite(x)):
                print(f"Non-finite state at burn-in iter {i} — stopping.")
                break

            if debug and i % 100 == 0:
                print(f"[burn-in {i:4d}]  U={U_x:.4e}  alpha={alpha:.3f}"
                      f"  L={L:.4e}  eta=1/L={1/L:.4e}"
                      f"  acc_rate={burn_accepts/(i+1):.3f}")

        L_frozen = L
        print(f"Burn-in done.  accept_rate={burn_accepts/n_burn:.4f}  "
              f"final L={L_frozen:.4e}  eta=1/L={1/L_frozen:.4e}")

        # ---------- Phase 2: averaging / sample collection ----------
        n_avg   = int(self.num_samples) * int(self.thin)
        samples_kept = []
        energies     = []
        accept_trace = np.zeros(n_avg, dtype=float)
        L_trace      = np.zeros(n_avg, dtype=float)   # stored as eta=1/L
        avg_accepts  = 0

        for i in range(n_avg):
            x, U_x, g_x, alpha, accepted = self._mala_step(x, U_x, g_x, L, y)

            if freeze_L_after_burnin:
                # hold L fixed at the burn-in value: step size is stable,
                # energy trace should be stationary, samples are comparable
                pass
            else:
                # continue gentle 1.1 adaptation
                if alpha < 0.574:
                    L *= 1.1
                else:
                    L /= 1.1
                L = float(np.clip(L, 1e-3, 1e15))

            avg_accepts += int(accepted)
            accept_trace[i] = float(accepted)
            L_trace[i]      = 1.0 / L
            energies.append(U_x)

            if not np.all(np.isfinite(x)):
                print(f"Non-finite state at averaging iter {i} — stopping.")
                break

            if debug and i % 100 == 0:
                print(f"[avg    {i:4d}]  U={U_x:.4e}  alpha={alpha:.3f}"
                      f"  L={L:.4e}  acc_rate={avg_accepts/(i+1):.3f}")

            # collect with thinning
            if i % self.thin == 0:
                samples_kept.append(np.copy(x))

        print(f"Averaging done. accept_rate={avg_accepts/n_avg:.4f}  "
              f"samples_collected={len(samples_kept)}")

        if len(samples_kept) == 0:
            print("Warning: no samples kept; returning last state.")
            samples_kept = [np.copy(x)]

        return samples_kept, energies, L_trace, accept_trace
    
    def compute_mmse_estimate(self, samples):
        """
        Compute MMSE estimate = mean over MALA posterior samples.
        """
       
        try:
            arr = np.stack(samples, axis=0)   # shape (N, H, W)
        except Exception as e:
            print("Failed to stack samples:", e)
            print("Sample shapes:", [np.shape(s) for s in samples])
            arr = np.expand_dims(np.array(samples[-1]), axis=0)

        return np.mean(arr, axis=0)
    
        # x_mmse = np.mean(arr, axis=0)
        # x_mmse = np.nan_to_num(x_mmse)
        # denom = x_mmse.max() - x_mmse.min()
        # if denom > 0:
        #     normalized_mmse = (x_mmse - x_mmse.min()) / denom
        # else:
        #     normalized_mmse = x_mmse
        # return normalized_mmse
    
    # def test_adjoint_A(est, H, W):
    #     x = np.random.randn(H, W) + 1j*np.random.randn(H, W)
    #     y = np.random.randn(H, W) + 1j*np.random.randn(H, W)
    #     lhs = np.vdot(est.A(x), y)
    #     rhs = np.vdot(x, est.A_adj(y))
    #     print("A adjoint diff:", lhs - rhs)

    # def test_grad_div_adjoint(est, H, W):
    #     x = np.random.randn(H, W)
    #     p = np.random.randn(H, W)
    #     q = np.random.randn(H, W)
    #     gx, gy = est.finite_diff_gradient(x)
    #     lhs = np.sum(gx * p + gy * q)
    #     rhs = -np.sum(x * est.divergence(p, q))
    #     print("grad-div diff:", lhs - rhs)

    # def test_full_gradient(est, H, W, y):
    #     x = np.random.randn(H, W)
    #     d = np.random.randn(H, W)
    #     d /= np.linalg.norm(d)  # normalize direction
    #     eps = 1e-6
    #     g_analytic = est.data_fidelity_gradient(x, y) + est.lambda_ * est.huber_tv_subgradient(x)
    #     ana = np.sum(g_analytic * d)
    #     Eplus  = est.compute_loss(x + eps*d, y)
    #     Eminus = est.compute_loss(x - eps*d, y)
    #     num = (Eplus - Eminus) / (2*eps)
    #     print("full grad diff:", ana - num, "relative:", abs(ana-num)/max(1e-12, abs(num)))