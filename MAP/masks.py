import numpy as np

class SimpleMask:
    def __init__(self, step, seed=None):
        ''' 
        Parameters:
        - step: how often to mask out a row/column (3 means mask out every 3rd)
        - seed: random seed
        '''

        self.step = step
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def mask_columns(self, x):
        """
        Create a mask that masks out every 'step'-th column.
 
        Parameters:
        - x: (n,m) matrix 
        - step: how often to mask out a column (e.g., 4 masks out every 4th)

        Returns:
        - masked_matrix: matrix with 0s in all masked columns
        """
        
        rows, cols = x.shape
        masked_matrix = np.copy(x)
        
        for j in range(1, cols, self.step):
            masked_matrix[:, j] = 0
         
        return masked_matrix
    
    def mask_rows(self, x):
        """
        Create a mask that masks out every 'step'-th rows.
 
        Parameters:
        - x: (n,m) matrix 
        - step: how often to mask out a row (e.g., 4 masks out every 4th)

        Returns:
        - masked_matrix: matrix with 0s in all masked rows
        """
        
        rows, cols = x.shape
        masked_matrix = np.copy(x)
        
        for j in range(1, rows, self.step):
            masked_matrix[j, :] = 0
            
        return masked_matrix