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
    
class CenteredBernoulliMask:
    '''
    Creates a matrix of 1s and 0s randomly, with the center of the matrix being 1s
    Parameters:
        - p: probability of keeping a value (outside of center) 
        - center_fraction: fraction of image height/width to define the central unmasked region (e.g., 0.5 means keep center 50% x 50%)
        - seed: optional seed for reproducibility
    '''

    def __init__(self, p, center_fraction, seed=None):
        self.p = p
        self.seed = seed
        self.center_fraction = center_fraction
        if seed is not None:
            np.random.seed(seed)

    def generate(self, shape):
        '''
        - shape: tuple (n, m), the shape of the mask
        Returns:
        - mask: (n, m) ndarray of 0s and 1s
        '''

        height, width = shape
        mask = np.random.binomial(n=1, p=self.p, size=shape)

        # define the center region
        center_height = int(height * self.center_fraction)
        center_width = int(height * self.center_fraction)
        height_start = (height - center_height) // 2
        width_start = (width - center_width) // 2

        # set the values in the center region to 1
        mask[height_start:height_start + center_height, width_start:width_start + center_width] = 1

        return mask

class VariableDensityMask:
    def __init__(self, decay_type='gaussian', decay_param=0.5, seed=None):
        '''
        Parameters:
        - decay_type: 'gaussian' or 'polynomial'
        - decay_param:
            - For 'gaussian': standard deviation (sigma) as a fraction of image size
            - For 'polynomial': power of the decay (e.g., 2 = quadratic)
        - seed: optional seed for reproducibility
        '''

        self.decay_type = decay_type
        self.decay_param = decay_param
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def distance_from_center(self, shape):
        h, w = shape
        y, x = np.ogrid[:h, :w] # creates a grid mesh
        center_y, center_x = h // 2, w // 2
        return np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
    def probability_map(self, shape):
        distance = self.distance_from_center(shape)
        max_distance = np.max(distance)
        norm_distance = distance / max_distance # normalizes/outputs values between 0 and 1

        if self.decay_type == 'gaussian':
            sigma = self.decay_param
            prob = np.exp(-0.5 * (norm_distance/sigma)**2) # sigma or 2*sigma?
        elif self.decay_type == 'polynomial':
            power = self.decay_param
            prob = 1 / (1 + norm_distance**power)
        else:
            raise ValueError(f"Unsupported decay_type: {self.decay_type}")
        
        return prob # values from 0 to 1
    
    def generate(self, shape):
        '''
        Generates a variable density sampling mask
        Parameters:
        - shape: tuple(h,w)
        Returns:
        - mask: ndarray of 0s and 1s, sampled based on distance-decayed probabilities
        '''
        prob_map = self.probability_map(shape)
        return np.random.binomial(1, prob_map)
    
    