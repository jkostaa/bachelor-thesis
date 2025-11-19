import numpy as np

# from scipy.ndimage import sobel
# from skimage.filters import scharr_h, scharr_v

# from scipy.optimize import minimize

import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\kostanjsek\bachelor_project"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Implementation of the class


class MAPEstimator:
    """
    Inputs for the MAP estimation algorithm:

    x_init: initial guess for image x, 2D real or complex array, shape of the image
    y: complex-valued 2D array, shape (rows, cols), actual MRI measurement after under-sampling in F-domain
    M: sampling mask, shape (same as y), with 0 where data isn't sampled
    sigma: noise variance, positive scalar, float
    lambda_: regularization parameter. positive scalar, float
    eps: Huber threshold parameter, small positive scalar, float
    max_iters: number of gradient descent steps, int
    learning_rate: gradient descent step size, float
    return x

    """

    def __init__(self, M, sigma, lambda_, eps, learning_rate, max_iters):
        self.M = M
        self.lambda_ = lambda_
        self.sigma = sigma
        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.loss_history = []

    # Forward and adjoint operators


    def A(self, x):
        """
        Defined as: A = M * F
        Compute the composition of 2D discrete Fourier Transform + apply mask
        Returns a (complex) ndarray type
        """

        #x = np.clip(x, -1e6, 1e6)  # avoiding large numbers
        # alt: x = x / np.max(np.abs(x))  # normalize to max=1, if max != 0

        # self.M * np.fft.fft2(x) / np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))
        return self.M * np.fft.fft2(x, norm='ortho') 

    def A_adj(self, k_residual):
        """
        Compute the adjoint of A
        """
        return np.fft.ifft2(self.M * k_residual, norm='ortho')  

    def adjoint_test(self, shape):
        """
        Sanity check for <A x, y> ≈ <x, A^* y>.
        A: forward operator 
        A_adj: adjoint operator 
        shape: tuple

        prints abs error.
        """
        # rand test vectors
        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        y = np.random.randn(*self.A(x).shape) + 1j * np.random.randn(*self.A(x).shape)

        lhs = np.vdot(self.A(x), y)          # <A x, y>
        rhs = np.vdot(x, self.A_adj(y))      # <x, A* y>

        abs_err = np.abs(lhs - rhs)
        rel_err = abs_err / max(np.abs(lhs), 1e-14) # 1e-14 term there, just to avoid zero division

        # print(f"<A x, y>  = {lhs}")
        # print(f"<x, A* y> = {rhs}")
        # print(f"Absolute error: {abs_err:.3e}")
        # print(f"Relative error: {rel_err:.3e}")

        return abs_err, rel_err

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

        # #alternative implementation
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)

        # gx[:, :-1] = x[:, 1:] - x[:, :-1]
        # gx[:, -1]  = -x[:, -1] 
        # #grad_x[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2

        # gy[:-1, :] = x[1:, :] - x[:-1, :]
        # gy[-1, :]  = -x[-1, :]
        #grad_y[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2


        gx[:, :-1] = x[:, 1:] - x[:, :-1]   # horizontal (right - center)
        gy[:-1, :] = x[1:, :] - x[:-1, :]   # vertical (bottom - center)
        
        # grad_x = (
        #     np.roll(x, -1, axis=1) - x
        # )  # horizontal, shift columns 1 position to the left and subtract with x
        # grad_y = np.roll(x, -1, axis=0) - x  # vertical

        return gx, gy

    '''
    def gradient_magnitude(self, x):
        """
        Computes the magnitude of the gradient: g = sqrt(g_x^2 + g_y^2)
        """

        grad_x, grad_y = self.finite_diff_gradient(x)

        grad_x = np.clip(grad_x, -1e6, 1e6)  # clipping to avoid very large numbers
        grad_y = np.clip(grad_y, -1e6, 1e6)

        return np.sqrt(
            grad_x**2 + grad_y**2 + 1e-8
        )  # adding a small term to avoid division by zero
    '''

    def huber_penalty_function_grad(self, dx, dy):  # =huber weights
        """
        Computes the derivative of the Huber penalty function w.r.t. dx, dy.
        """
        t = np.sqrt(dx**2 + dy**2)
        w = np.where(t <= self.eps, 1.0 / self.eps, 1.0 / (t + 1e-12))
        grad_dx = w * dx
        grad_dy = w * dy
        return grad_dx, grad_dy
    
        # abs_dx = np.abs(dx)
        # abs_dy = np.abs(dy)

        # grad_dx = np.where(abs_dx >= self.eps, np.sign(dx), dx / self.eps,)
        # grad_dy = np.where(abs_dy >= self.eps, np.sign(dy), dy / self.eps)

        # return grad_dx, grad_dy 

    def divergence(self, px, py):
        """
        Computes the divergence of a vector field
        Input: 2 arrays (vector field)
        Output: 2D array (scalar)
        """

        # div_x = norm_grad_x - np.roll(norm_grad_x, 1, axis=1)
        # div_y = norm_grad_y - np.roll(norm_grad_y, 1, axis=0)

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

        '''
        def huber_penalty_function(t, eps):
            """
            Computes the Huber penalty function: quadratic for |t| <= eps, linear otherwise.
            t: array (differences, e.g. residuals)
            eps: scalar threshold
            """

            abs_t = np.abs(t)
            quad = (t**2) / (2 * eps)
            lin = abs_t - (eps / 2)
            return np.where(
                abs_t >= eps, lin, quad
            )  # condition if |t| >= eps; return lin, else return quad
        '''

        dx, dy = self.finite_diff_gradient(x)

        t = np.sqrt(dx**2 + dy**2)
        quad = (t**2) / (2 * self.eps)
        lin = np.abs(t) - (self.eps / 2) # np.abs(t) or just t
        tv = np.where(t <= self.eps, quad, lin)

        # dx = x[:,1:] - x[:,:-1] # horizontal (j+1 - j)... sub-optimal implementation (shape mismatch, no boundary)
        # dy = x[1:,:] - x[:-1,:] # vertical (i+1 - i)

        #tv_x = huber_penalty_function(dx, self.eps)
        #tv_y = huber_penalty_function(dy, self.eps)

        return tv.sum()

    # start of implementation of Huber-TV subgradient

    def huber_tv_subgradient(self, x):
        """
        Computes the subgradient of TV
        Return: subgradient of TV
        There is a divergence used at the end -> divergence maps back to scalar - same shape as x (image)
        """

        dx, dy = self.finite_diff_gradient(x)
        # mag = self.gradient_magnitude(x)
        grad_dx, grad_dy = self.huber_penalty_function_grad(dx, dy)
        # normalize - gradient direction * scalar weight -> gives a directional subgradient

        return -self.divergence(grad_dx, grad_dy)

    def compute_loss(self, x, y): # check maybe if data_term is correct (2* sigma or just sigma)
        '''
        Function to compute the loss (used later in the subgradient descent function
        to plot the loss over iterations)
        '''
        data_term = np.linalg.norm(self.A(x) - y) ** 2 / (2 * self.sigma**2)
        tv_term = self.lambda_ * self.huber_tv_2d(x)
        return data_term + tv_term

    def subgradient_descent(self, y, x_init=None):
        """
        Minimization function
        """
    
        self.grad_norm_history = []
        
        # x = np.zeros_like(np.fft.ifft2(y).real)

        if x_init is None:
            x = np.real(np.fft.ifft2(y)) # or np.abs()?
        else:
            x = np.array(x_init, dtype=float)


        for i in range(self.max_iters):
            # a = self.A(x) # for print check

            gradient_data = self.data_fidelity_gradient(x, y)
            gradient_tv = self.huber_tv_subgradient(x)
            # print(f"Iteration {i}: grad_y min={grad_y.min():.2e}, max={grad_y.max():.2e}, mean={grad_y.mean():.2e}")
            gradient = gradient_data + self.lambda_ * gradient_tv
            gradient = gradient.real

            grad_norm = np.linalg.norm(gradient)
            self.grad_norm_history.append(grad_norm)

            x -= self.learning_rate * gradient

            # print(f"Iteration {i}, update norm: {np.linalg.norm(self.learning_rate * gradient)}")

            # computing the loss
            loss = self.compute_loss(x, y)
            self.loss_history.append(loss)

            if i % 50 == 0:
                print(f"Iter {i}: Loss = {loss:.8f}")
                print(f"Iter {i}: Gradient = {grad_norm:.8f}")
                print("||A(x)-y||_2 =", np.linalg.norm(self.A(x) - y))
                print("data_term =", np.linalg.norm(self.A(x) - y)**2 / (2*self.sigma**2))
                print("tv_term =", self.lambda_ * self.huber_tv_2d(x))
                print("||grad_data|| =", np.linalg.norm(gradient_data))
                print("||grad_tv|| =", np.linalg.norm(gradient_tv))
                print("grad min/max:", gradient.min(), gradient.max())

        return x

