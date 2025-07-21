'''
This file is for running the MAP estimation on skimage.data:
    - Load images from skimage
    - Generate k-space with FFT
    - Simulate undersampling
    - Run the reconstruction
    - Display/compare results
'''

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from map_tv_minimize import subgradient_descent, compute_A

np.set_printoptions(threshold=np.inf)
#shepp_logan[:100, :100] = 0
#inds_r = np.arange(len(shepp_logan))
#inds_c = 4 * inds_r % len(shepp_logan)
#shepp_logan[inds_r, inds_c] = 1

shepp_logan = ski.data.shepp_logan_phantom()

def simple_column_mask(x, step, seed=None):
    """
    Create a mask that samples every 'step'-th column.

    Parameters:
    - x: (n,m) matrix
    - step: how often to keep a column (e.g., 4 keeps every 4th)

    Returns:
    - masked_matrix: matrix with 0s in all masked columns and 
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    rows, cols = x.shape
    masked_matrix = np.copy(x)
    
    for j in range(1, cols, step):
        masked_matrix[:, j] = 0
    	
    return masked_matrix


masked_shepp_logan = simple_column_mask(shepp_logan, 2)

k_space = compute_A(shepp_logan, masked_shepp_logan)

print(k_space)

plt.figure()
plt.imshow(shepp_logan, cmap='gray')
plt.title('Shepp-Logan phantom')
#print(shepp_logan[134, 201])
#print(np.min(shepp_logan), np.max(shepp_logan), np.mean(shepp_logan))

plt.figure()
plt.imshow(masked_shepp_logan, cmap='gray')
#print(simple_column_mask(shepp_logan, 4))

plt.show()
