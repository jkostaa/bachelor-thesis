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

def simple_column_mask(shape, stride):
    '''
    Creates a mask that samples 
    '''

    pass
#M = create_random_mask(img.shape, acceleration=4)
#k_space = compute_A(shepp_logan, M)

plt.imshow(shepp_logan, cmap='gray')
plt.title('Shepp-Logan phantom')
print(shepp_logan[134, 201])
print(np.min(shepp_logan), np.max(shepp_logan), np.mean(shepp_logan))
plt.show()
