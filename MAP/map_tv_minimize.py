import numpy as np

# from scipy.ndimage import sobel
# from skimage.filters import scharr_h, scharr_v
# from scipy.ndimage import gaussian_filter

# from scipy.optimize import minimize

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

    def __init__(self, M, sigma, lambda_, eps, learning_rate=0.1, max_iters=100):
        self.M = M
        self.lambda_ = lambda_
        self.sigma = sigma
        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.loss_history = []

    def A(self, x):
        """
        Defined as: A = M * F
        Compute the composition of 2D discrete Fourier Transform + apply mask
        Returns a (complex) ndarray type
        """

        x = np.clip(x, -1e6, 1e6)  # avoiding large numbers
        # alt: x = x / np.max(np.abs(x))  # normalize to max=1, if max != 0

        return self.M * np.fft.fft2(x)

    def A_adj(self, k_residual):
        """
        Compute the adjoint of A
        """
        return np.fft.ifft2(self.M * k_residual).real  # added .real

    def data_fidelity_gradient(self, x, y):
        """
        k_residual: Ax - y <-> M * F(x) - y ... error between predicted & measured k-space data, has dimensions of k-space(=residual)
        Returns the gradient (derived after x) of the data fidelity term (1/2*sigma * ||Ax - y||^2)
        """

        residual = self.A(x) - y
        grad = self.A_adj(residual) / self.sigma**2
        return grad

    def finite_diff_gradient(self, x):  # we are assuming that x is a 2D image
        """
        Computes the forward finite differences
        Input: 2D array
        Output: 2 arrays
        Returns the horizontal and vertical gradient
        Question: current implementation is using wrap-around, is this reasonable?
        """

        """
        #alternative implementation
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(x)
        #grad_x[:, :-1] = x[:, 1:] - x[:, :-1]
        grad_x[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2

        #grad_y[:-1, :] = x[1:, :] - x[:-1, :]
        grad_y[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
        """

        grad_x = (
            np.roll(x, -1, axis=1) - x
        )  # horizontal, shift columns 1 position to the left and subtract with x
        grad_y = np.roll(x, -1, axis=0) - x  # vertical

        # grad_x = sobel(x, axis=1, mode='reflect')  # Horizontal gradient
        # grad_y = sobel(x, axis=0, mode='reflect')  # Vertical gradient

        # grad_x = scharr_h(x)
        # grad_y = scharr_v(x)

        # grad_x = gaussian_filter(x, sigma=1, order=[0,1])
        # grad_y = gaussian_filter(x, sigma=1, order=[1,0])

        return grad_x, grad_y

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

    def huber_penalty_function_grad(self, x):  # =huber weights
        """
        Computes the derivative of the Huber penalty function
        """

        mag = self.gradient_magnitude(x)
        return np.where(
            mag >= self.eps, 1.0, mag / self.eps
        )  # mag always >0, so no need for np.abs(mag) >= eps, np.sign(mag), ...

    def divergence(self, norm_grad_x, norm_grad_y):
        """
        Computes the divergence of a vector field
        Input: 2 arrays (vector field)
        Output: 2D array (scalar)
        """

        div_x = norm_grad_x - np.roll(norm_grad_x, 1, axis=1)
        div_y = norm_grad_y - np.roll(norm_grad_y, 1, axis=0)
        return div_x + div_y

    def huber_tv_2d(self, x):
        """
        A function that computes the Huber total variation of a 2D array (resembling a picture made up of n x m pixels)

        Parameters:
        x: ndarray of shape (n,m) - the input image -> approximated x
        eps: float, threshold value (threshold between quadratic and linear region)
        dx: horizontal differences (n, m-1)
        dy: vertical differences (n-1, m)
        """

        def huber_penalty_function(eps, t):
            """
            Computes the Huber penalty function
            """

            abs_t = np.abs(t)
            quad = (t**2) / (2 * eps)
            lin = abs_t - (eps / 2)
            return np.where(
                abs_t >= eps, lin, quad
            )  # condition if |t| >= eps; return lin, else return quad

        # Compute the finite differences

        dx, dy = self.finite_diff_gradient(x)

        # dx = x[:,1:] - x[:,:-1] # horizontal (j+1 - j)... sub-optimal implementation (shape mismatch, no boundary)
        # dy = x[1:,:] - x[:-1,:] # vertical (i+1 - i)

        tv_x = huber_penalty_function(dx, self.eps)
        tv_y = huber_penalty_function(dy, self.eps)

        return tv_x.sum() + tv_y.sum()

    # start of implementation of Huber-TV subgradient

    def huber_tv_subgradient(self, x):
        """
        Computes the subgradient of TV
        Return: subgradient of TV
        There is a divergence used at the end -> divergence maps back to scalar - same shape as x (image)
        """

        grad_x, grad_y = self.finite_diff_gradient(x)
        # mag = self.gradient_magnitude(x)
        weights = self.huber_penalty_function_grad(x)

        # normalize - gradient direction * scalar weight -> gives a directional subgradient
        norm_grad_x = weights * grad_x
        norm_grad_y = weights * grad_y

        return -self.divergence(norm_grad_x, norm_grad_y)

    def compute_loss(self, x, y):
        '''
        Function to compute the loss (used later in the subgradient descent function
        to plot the loss over iterations)
        '''
        data_term = np.linalg.norm(self.A(x) - y) ** 2 / (2 * self.sigma**2)
        tv_term = self.lambda_ * self.huber_tv_2d(x)
        return data_term  + tv_term

    def subgradient_descent(self, y):
        """
        Minimization function
        """
        # print("Starting subgradient descent...")
        # x = x_init if x_init is not None else np.zeros_like(np.fft.ifft2(y).real)
        x = np.zeros_like(np.fft.ifft2(y).real)

        # x = np.fft.ifft2(self.M * np.fft.fft2(x_init)).real
        # x = x_init.copy()
        for i in range(self.max_iters):
            # a = self.A(x) # for print check

            gradient_data = self.data_fidelity_gradient(x, y)
            gradient_tv = self.huber_tv_subgradient(x)
            # print(f"Iteration {i}: grad_y min={grad_y.min():.2e}, max={grad_y.max():.2e}, mean={grad_y.mean():.2e}")
            gradient = gradient_data + self.lambda_ * gradient_tv

            # noise = np.random.randn(*x.shape) # Langevin part
            x -= self.learning_rate * gradient
            # x += np.sqrt(2 * self.learning_rate) * noise # Langevin part
            # print(f"Iteration {i}, update norm: {np.linalg.norm(self.learning_rate * gradient)}")

            # computing the loss
            loss = self.compute_loss(x, y)
            self.loss_history.append(loss)

            if i % 10 == 0:
                print(f"Iter {i}: Loss = {loss:.8f}")

        return x

    def langevin_sampling(self, y, num_samples=100, burn_in=150, sample_every=10):
        """
        Parameters:
        - y: ndarray, undersampled k-space data
        - num_samples: int, how many samples to return after burn-in
        - burn_in: int, number of initial steps to discard before collecting (tries to avoid bias due to x_init)
        - sample_every: int, interval between storing samples
        """

        x = np.ones_like(
            np.fft.ifft2(y).real
        )  # initialized with zero-filled inverse FFT
        samples = []

        for i in range(burn_in + num_samples * sample_every):
            gradient_data = self.data_fidelity_gradient(x, y)
            gradient_tv = self.huber_tv_subgradient(x)
            gradient = gradient_data + self.lambda_ * gradient_tv

            noise = np.random.randn(*x.shape)
            x -= self.learning_rate * gradient
            x += np.sqrt(2 * self.learning_rate) * noise

            if i >= burn_in and (i - burn_in) % sample_every == 0:
                samples.append(np.copy(x))

        return samples


"""
for iteration in range(max_iters):
# Inputs: y (measured k-space), M, lambda, sigma, eps (huber threshold), step size, max_iters
    # Data fidelity gradient
    residual = A(x) - y
    grad_data = A_adjoint(residual) / sigma^2

    # Compute Huber TV subgradient
    grad_tv = huber_tv_subgradient(x, delta)

    # Total subgradient
    subgrad = grad_data + lambda * grad_tv

    # Step update
    x -= learning_rate * (data_fidelity_grad + λ * tv_subgrad)
"""


"""
def map_tv_minimize(x, y, lambda_tv=0.1, max_iter=100):
    
    Minimize the total variation of the image x with respect to y.
    
    Parameters:
    - x: Input image (numpy array).
    - y: Target image (numpy array).
    - lambda_tv: Regularization parameter for total variation.
    - max_iter: Maximum number of iterations for optimization.
    
    Returns:
    - x_min: Optimized image after minimizing total variation.
    
    # Placeholder for the optimization logic
    # This is where you would implement the actual minimization algorithm
    x_min = x  # For now, just return the input image as a placeholder
    
    return x_min
"""


"""
# Example usage:
if __name__ == "__main__":
    x = np.random.rand(100, 100)  # Example input image
    y = np.random.rand(100, 100)  # Example target image
    lambda_tv = 0.1
    max_iter = 100
    
    x_min = map_tv_minimize(x, y, lambda_tv, max_iter)
    print("Optimized image shape:", x_min.shape)

"""
