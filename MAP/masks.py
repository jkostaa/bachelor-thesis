import numpy as np


class SimpleMask:
    def __init__(self, step):
        """
        Parameters:
        - step: how often to mask out a row/column (3 means mask out every 3rd)
        """

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
        # masked_matrix = np.copy(x)

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
        # masked_matrix = np.copy(x)

        for j in range(1, rows, self.step):
            M[j, :] = 0

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

class PseudoRandomColumnMask:
    def __init__(self, shape, acceleration, seed=None):
        """
        Parameters:
        - shape: tuple (h, w)
        - acceleration: int, undersampling factor (e.g., 4 or 8)
        - seed: random seed for reproducibility
        """

        assert acceleration in [2, 4, 6, 8], "Only acceleration factors 2, 4, 6 and 8 are supported."

        self.shape = shape
        self.acceleration = acceleration
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

        # indices outside the center region
        candidate_cols = list(range(0, center_start)) + list(range(center_end, width))

        if num_random_cols > 0:
            random_cols = np.random.choice(candidate_cols, size=num_random_cols, replace=False)
            mask[:, random_cols] = 1
        
        return mask
    
    def get_mask(self):
        return self.mask