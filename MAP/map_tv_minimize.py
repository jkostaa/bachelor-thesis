import numpy as np
# from scipy.optimize import minimize   

# Implementation of the class

class MAPEstimator:
    '''
    Inputs for the MAP estimation algorithm:

    x_init: initial guess for image x, 2D real or complex array, shape of the image -> why do i need this?
    y: complex-valued 2D array, shape (rows, cols), actual MRI measurement after under-sampling in F-domain
    M: sampling mask, shape (same as y), with 0 where data isn't sampled
    sigma: noise variance, positive scalar, float
    lambda_: regularization parameter. positive scalar, float
    eps: Huber threshold parameter, small positive scalar, float
    max_iters: number of gradient descent steps, int 
    learning_rate: gradient descent step size, float
    return x

    x_init and y as input parameters?
    '''
    def __init__(self, M, sigma, lambda_, eps, learning_rate=0.1, max_iters=100):
        self.M = M
        self.lambda_ = lambda_
        self.sigma = sigma
        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        pass

    def A(self, x):
        '''
        Defined as: A = M * F
        Compute the composition of 2D discrete Fourier Transform + apply mask
        Returns a (complex) ndarray type
        '''

        return self.M * np.fft.fft2(x)

    def A_adj(self, k_residual):
        '''
        Compute the adjoint of A
        '''
        return np.fft.ifft2(self.M * k_residual).real # added .real

    def data_fidelity_gradient(self, x, y):
        '''
        k_residual: Ax - y <-> M * F(x) - y ... error between predicted & measured k-space data, has dimensions of k-space(=residual)
        Returns the gradient (derived after x) of the data fidelity term (1/2*sigma * ||Ax - y||^2)
        '''

        residual = self.A(x) - y
        grad = self.A_adj(residual) /self.sigma**2
        return grad
    
    def finite_diff_gradient(self, x): # we are assuming that x is a 2D image
        '''
        Computes the forward finite differences
        Input: 2D array
        Output: 2 arrays
        Returns the horizontal and vertical gradient
        Question: current implementation is using wrap-around, is this reasonable?
        '''

        '''
        alternative implementation
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:, :-1] = x[:, 1:] - x[:, :-1]
        gy[:-1, :] = x[1:, :] - x[:-1, :]
        '''
        
        grad_x = np.roll(x, -1, axis=1) - x # horizontal, shift columns 1 position to the left and subtract with x
        grad_y = np.roll(x, -1, axis=0) - x # vertical

        return grad_x, grad_y

    def gradient_magnitude(self, x):
        '''
        Computes the magnitude of the gradient: g = sqrt(g_x^2 + g_y^2)
        '''

        grad_x, grad_y = self.finite_diff_gradient(x)
        return np.sqrt(grad_x**2 + grad_y**2 + 1e-8) # adding a small term to avoid division by zero

    def huber_penalty_function_grad(self, x, eps): # =huber weights
        '''
        Computes the derivative of the Huber penalty function
        '''

        mag = self.gradient_magnitude(x)
        return np.where(mag >= eps, 1.0, mag/eps)  # mag always >0, so no need for np.abs(mag) >= eps, np.sign(mag), ...

    def divergence(self, norm_grad_x, norm_grad_y):
        '''
        Computes the divergence of a vector field
        Input: 2 arrays (vector field)
        Output: 2D array (scalar)
        '''
        
        div_x = norm_grad_x - np.roll(norm_grad_x, 1, axis=1)
        div_y = norm_grad_y - np.roll(norm_grad_y, 1, axis=0)
        return div_x + div_y

def huber_tv_2d(x, eps): 
    '''
    A function that computes the Huber total variation of a 2D array (resembling a picture made up of n x m pixels)

    Parameters:
    x: ndarray of shape (n,m) - the input image -> approximated x
    eps: float, threshold value (threshold between quadratic and linear region)
    dx: horizontal differences (n, m-1)
    dy: vertical differences (n-1, m)
    '''
    def huber_penalty_function(t, eps): # t and eps both free variables?
        '''
        Computes the Huber penalty function
        '''
        
        abs_t = np.abs(t)
        quad = (t**2) / (2*eps)
        lin = abs_t - (eps/2)
        return np.where(abs_t >= eps, lin, quad) # condition if |t| >= eps; return lin, else return quad 
    
    # Compute the finite differences

    dx, dy = finite_diff_gradient(x)

    #dx = x[:,1:] - x[:,:-1] # horizontal (j+1 - j)... sub-optimal implementation (shape mismatch, no boundary)
    #dy = x[1:,:] - x[:-1,:] # vertical (i+1 - i) 

    tv_x = huber_penalty_function(dx, eps).sum()
    tv_y = huber_penalty_function(dy, eps).sum()

    return tv_x + tv_y

# start of implementation of Huber-TV subgradient

def huber_tv_subgradient(x, eps):
    '''
    Computes the subgradient of TV
    Return: subgradient of TV
    There is a divergence used at the end -> divergence maps back to scalar - same shape as x (image)
    '''

    grad_x, grad_y = finite_diff_gradient(x)
    mag = gradient_magnitude(x)
    weights = huber_penalty_function_grad(x, eps)

    # normalize - gradient direction * scalar weight -> gives a directional subgradient
    norm_grad_x = weights * grad_x
    norm_grad_y = weights * grad_y

    return -divergence(norm_grad_x, norm_grad_y) 

def subgradient_descent(x_init, y, M, lambda_, eps, sigma, learning_rate, max_iters):
    '''
    Minimization function
    '''

    x = x_init.copy()
    for i in range(max_iters):
        # Ax = compute_A(x,M) # forward model
        # z = Ax - y # residual
        gradient_data = data_fidelity_gradient(x, y, M, sigma)
        gradient_tv = huber_tv_subgradient(x, eps)
        gradient = gradient_data + lambda_ * gradient_tv
        x -= learning_rate * gradient
    
    return x


'''
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
'''


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


'''
# Example usage:
if __name__ == "__main__":
    x = np.random.rand(100, 100)  # Example input image
    y = np.random.rand(100, 100)  # Example target image
    lambda_tv = 0.1
    max_iter = 100
    
    x_min = map_tv_minimize(x, y, lambda_tv, max_iter)
    print("Optimized image shape:", x_min.shape)

'''

