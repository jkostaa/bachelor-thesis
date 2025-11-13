import numpy as np
from sympy import I

class SimpleMask: # not relevant for thesis
    def __init__(self, step):
        """
        Parameters:
        - step: how often to sample a row/column (3 means sample every 3rd)
        """

        self.step = step

    def mask_columns(self, shape):
        """
        Create a mask that samples every 'step'-th column.

        Parameters:
        - x: (n,m) matrix

        Returns:
        - M: matrix with 1s in all sampled columns
        """

        rows, cols = shape
        M = np.zeros((rows, cols), dtype=np.float32) 
        # masked_matrix = np.copy(x)

        for i in range(0, cols, self.step):
            M[:, i] = 1

        return M

    def mask_rows(self, shape):
        """
        Create a mask that samples every 'step'-th rows.

        Parameters:
        - x: (n,m) matrix

        Returns:
        - M: matrix with 1s in all sampled rows
        """

        rows, cols = shape
        M = np.zeros((rows, cols), dtype=np.float32) # use np.zeros for masking
        # masked_matrix = np.copy(x)

        for j in range(0, rows, self.step):
            M[j, :] = 1 # use value 0 for masking

        return M

class SimpleMask2D: # not relevant for thesis
    def __init__(self, row_step=None, col_step=None):

        """
        Parameters:
        - row_step: sampling interval for rows (None = no row masking)
        - col_step: sampling interval for columns (None = no column masking)
        """

        self.row_step = row_step
        self.col_step = col_step

    def get_mask(self, shape):
        """
        Create a 2D sampling mask (both rows + columns)
        
        Parameters:
        - x: (n, m) matrix
        
        Returns:
        - M: mask with 1s at sampled positions, 0s elsewhere
        """

        rows, cols = shape
        M = np.zeros((rows, cols), dtype=np.float32)

        # Sample rows
        if self.row_step is not None:
            for i in range(0, rows, self.row_step):
                M[i, :] = 1

        # Sample columns
        if self.col_step is not None:
            for j in range(0, cols, self.col_step):
                M[:, j] = 1

        return M

class BernoulliMask:
    """
    Creates a matrix of 1s and 0s randomly
    Parameters:
    - p: probability of keeping a value (from 0 to 1)
    - seed: optional seed for reproducibility
    """

    def __init__(self, p, seed=None):
        self.p = p
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate(self, shape):
        """
        - shape: tuple (n, m), the shape of the mask
        Returns:
        - mask: (n, m) ndarray of 0s and 1s
        """
        return np.random.binomial(n=1, p=self.p, size=shape)


class CenteredBernoulliMask:
    """
    Creates a matrix of 1s and 0s randomly, with the center of the matrix being 1s
    Parameters:
        - p: probability of keeping a value (outside of center)
        - center_fraction: fraction of image height/width to define the central unmasked region (e.g., 0.5 means keep center 50% x 50%)
        - seed: optional seed for reproducibility
    """

    def __init__(self, p, center_fraction, seed=None):
        self.p = p
        self.seed = seed
        self.center_fraction = center_fraction
        if seed is not None:
            np.random.seed(seed)

    def generate(self, shape):
        """
        - shape: tuple (n, m), the shape of the mask
        Returns:
        - mask: (n, m) ndarray of 0s and 1s
        """

        height, width = shape
        mask = np.random.binomial(n=1, p=self.p, size=shape)

        # define the center region
        center_height = int(height * self.center_fraction)
        center_width = int(height * self.center_fraction)
        height_start = (height - center_height) // 2
        width_start = (width - center_width) // 2

        # set the values in the center region to 1
        mask[
            height_start : height_start + center_height,
            width_start : width_start + center_width,
        ] = 1

        return mask


class VariableDensityMask:
    def __init__(self, decay_type="gaussian", decay_param=0.5, seed=None):
        """
        Parameters:
        - decay_type: 'gaussian' or 'polynomial'
        - decay_param:
            - For 'gaussian': standard deviation (sigma) as a fraction of image size
            - For 'polynomial': power of the decay (e.g., 2 = quadratic)
        - seed: optional seed for reproducibility
        """

        self.decay_type = decay_type
        self.decay_param = decay_param
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def distance_from_center(self, shape):
        h, w = shape
        y, x = np.ogrid[:h, :w]  # creates a grid mesh
        center_y, center_x = h // 2, w // 2
        return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    def probability_map(self, shape):
        distance = self.distance_from_center(shape)
        max_distance = np.max(distance)
        norm_distance = (
            distance / max_distance
        )  # normalizes/outputs values between 0 and 1

        if self.decay_type == "gaussian":
            sigma = self.decay_param
            prob = np.exp(-0.5 * (norm_distance / sigma) ** 2)  # sigma or 2*sigma?
        elif self.decay_type == "polynomial":
            power = self.decay_param
            prob = 1 / (1 + norm_distance**power)
        else:
            raise ValueError(f"Unsupported decay_type: {self.decay_type}")

        return prob  # values from 0 to 1

    def generate(self, shape):
        """
        Generates a variable density sampling mask
        Parameters:
        - shape: tuple(h,w)
        Returns:
        - mask: ndarray of 0s and 1s, sampled based on distance-decayed probabilities
        """
        prob_map = self.probability_map(shape)
        return np.random.binomial(1, prob_map)

class UniformColumnMask:
    def __init__(self, shape, acceleration, seed=None):
        """
        Uniform (non-random) 1D Cartesian undersampling mask.
        
        Parameters:
        - shape: tuple (h, w)
        - acceleration: int, undersampling factor (e.g., 2, 4, 6, 8)
        - seed: unused (kept for compatibility)
        """
        assert acceleration in [2, 4, 6, 8], "Only acceleration factors 2, 4, 6, and 8 are supported."

        self.shape = shape
        self.acceleration = acceleration
        self.mask = self._create_mask()

    def _create_mask(self):
        height, width = self.shape

        # center fraction definitions (same as before)
        if self.acceleration == 4:
            center_fraction = 0.08
        elif self.acceleration == 8:
            center_fraction = 0.04
        elif self.acceleration == 6:
            center_fraction = 0.06
        elif self.acceleration == 2:
            center_fraction = 0.10

        center_cols = int(round(width * center_fraction))
        if center_cols % 2 == 0:
            center_cols += 1

        mask = np.zeros((height, width), dtype=np.float32)

        # Fully sampled center region
        center_start = width // 2 - center_cols // 2
        center_end = center_start + center_cols
        mask[:, center_start:center_end] = 1

        # Uniformly spaced outer columns
        step = self.acceleration  # every R-th column
        # Choose offset so that sampling is symmetric around center
        offset = (width // 2) % step

        for col in range(0, width, step):
            if col < center_start or col >= center_end:
                mask[:, col] = 1

        return mask

    def get_mask(self):
        return self.mask

class PseudoRandomColumnMask:
    def __init__(self, shape, acceleration, lam, seed=None):
        """
        Parameters:
        - shape: tuple (h, w)
        - acceleration: int, undersampling factor (e.g., 2, 4, 6 or 8)
        - lam: float, decay parameter (e.g. 4)
        - seed: random seed for reproducibility
        """

        assert acceleration in [2, 4, 6, 8], "Only acceleration factors 2, 4, 6 and 8 are supported."

        self.shape = shape
        self.acceleration = acceleration
        self.lam = lam # controls steepness of decay
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.mask = self._create_mask()

    def _create_mask(self):
        height, width = self.shape

        if self.acceleration == 4:
            center_fraction = 0.08
        elif self.acceleration == 8:
            center_fraction = 0.04
        elif self.acceleration == 6:
            center_fraction = 0.06
        elif self.acceleration == 2:
            center_fraction = 0.1   

        center_cols = int(round(width * center_fraction)) # number of fully sampled center columns
        if center_cols % 2 == 0:
            center_cols += 1  # odd for symmetry

        mask = np.zeros((height, width), dtype=np.float32) # init

        # fill in the center region (centred horizontally)
        center_start = width // 2 - center_cols // 2
        center_end = center_start + center_cols
        mask[:, center_start:center_end] = 1

        total_sampled_cols = int(round(width / self.acceleration))

        # Number of additional columns to randomly sample
        num_random_cols = total_sampled_cols - center_cols
        if num_random_cols < 0:
            num_random_cols = 0 
            return mask

        distances = np.abs(np.arange(width) - width / 2)
        distances = distances / np.max(distances)  # normalize [0, 1]

        #probs = np.exp(-self.lam * distances) # exponential decay
        probs = np.exp(-self.lam * distances**2) # gaussian decay

        probs[center_start:center_end] = 0
        probs = probs / np.sum(probs)

        candidate_cols = np.arange(width)
        random_cols = np.random.choice(candidate_cols, size=num_random_cols, replace=False, p=probs)
        mask[:, random_cols] = 1
        
        return mask
    
    def get_mask(self):
        return self.mask
    

class RadialMask:
    def __init__(self, shape, num_spokes, center_fraction=0.05):
        
        """
        Create a binary radial undersampling mask.
        
        Parameters
        ----------
        shape : tuple (H, W)
            Mask size (assumed square or nearly square).
        num_spokes : int
            Number of radial lines (spokes) through k-space center.
        center_fraction : float
            Fully sampled central region (radius fraction of image size).
        """

        self.shape = shape
        self.num_spokes = num_spokes
        self.center_fraction = center_fraction

    def generate(self):
        H, W = self.shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        
        # radial coordinates (centered)
        x = X - cx
        y = Y - cy

        mask = np.zeros((H, W), dtype=np.float32)

        # Angles for evenly spaced spokes

        golden_angle = np.pi / ( (1 + np.sqrt(5)) / 2 )  # ≈ 111.25°
        angles = np.mod(np.arange(self.num_spokes) * golden_angle, np.pi)
        #angles = np.linspace(0, np.pi, self.num_spokes, endpoint=False)

        # Create spokes
        for theta in angles:
            # parametric line through center
            r = x * np.cos(theta) + y * np.sin(theta)
            mask[np.abs(r) < 0.5] = 1  # line thickness ~1 pixel

        # Add fully-sampled low-frequency circle
        R = np.sqrt(x**2 + y**2)
        mask[R < (min(H, W) * self.center_fraction / 2)] = 1

        return mask
