import numpy as np
# from scipy.optimize import minimize   

'''
def least_squares_solver(A, y):
    
    Computes the least squares problem: minimize ||Ax - y||^2 in the 2-norm
    A: numpy.ndarray, shape (m, n), forward model
    y: numpy.ndarray, shape (m,), measurement 
    x: solution vector (unmeasured data, desired image)
    
    x = np.linalg.lstsq(A, y, rcond=None)

    # possibility to return residuals
    return x
'''

def A(x, M):
    '''
    Compute the composition of 2D discrete Fourier Transform + apply mask
    Returns a (complex) ndarray type
    '''

    return M * np.fft.fft2(x) 

def A_adj(k_residual, M):
    '''
    Compute the adjoint of A
    '''
    return np.fft.ifft2(M * k_residual)


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
    dx = x[:,1:] - x[:,:-1] # horizontal (j+1 - j)
    dy = x[1:,:] - x[:-1,:] # vertical (i+1 - i)

    tv_x = huber_penalty_function(dx, eps).sum()
    tv_y = huber_penalty_function(dy, eps).sum()

    return tv_x + tv_y

def huber_penalty_function_grad(t, eps): # optional?
    '''
    Computes the derivative of the Huber penalty function
    '''
    return np.where(np.abs(t) >= eps, np.sign(t), t/eps)



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
    # Further processing can be done with x_min

    # Note: The actual implementation of the optimization algorithm is not provided.
    # You would need to implement the logic for minimizing total variation here.    

    # For example, you could use gradient descent or another optimization method.
'''

