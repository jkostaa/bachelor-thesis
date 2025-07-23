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