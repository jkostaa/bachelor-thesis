import numpy as np

class SimpleMask:
    def __init__(self, step):
        ''' 
        Parameters:
        - step: how often to mask out a row/column (3 means mask out every 3rd)
        '''

        self.step = step
        
    def mask_columns(self, x):
        """
        Create a mask that masks out every 'step'-th column.
 
        Parameters:
        - x: (n,m) matrix 

        Returns:
        - M: matrix with 0s in all masked columns
        """
        
        rows, cols = x.shape
        M = np.ones((rows, cols), dtype=np.float32)
        #masked_matrix = np.copy(x)
        
        for j in range(1, cols, self.step):
            M[:, j] = 0
         
        return M
    
    def mask_rows(self, x):
        """
        Create a mask that masks out every 'step'-th rows.
 
        Parameters:
        - x: (n,m) matrix 

        Returns:
        - M: matrix with 0s in all masked rows
        """
        
        rows, cols = x.shape
        M = np.ones((rows, cols), dtype=np.float32)
        #masked_matrix = np.copy(x)
        
        for j in range(1, rows, self.step):
            M[j, :] = 0
            
        return M    
    

class BernoulliMask:
    '''
    Creates a matrix of 1s and 0s randomly
    Parameters:
    - p: probability of keeping a value (from 0 to 1)
    - seed: optional seed for reproducibility
    '''

    def __init__(self, p, seed=None):
        self.p = p
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
    def generate(self, shape):
        '''
        - shape: tuple (n, m), the shape of the mask
        Returns:
        - mask: (n, m) ndarray of 0s and 1s
        '''
        return np.random.binomial(n=1, p=self.p, size=shape)