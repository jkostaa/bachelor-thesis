'''
This file is for running the MAP estimation on skimage.data:
    - Load images from skimage
    - Generates k-space with FFT
    - Simulate undersampling
    - Runs the reconstruction
    - Display/compare results
'''

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from map_tv_minimize import MAPEstimator
from masks import SimpleMask

np.set_printoptions(threshold=np.inf)

# load the image
shepp_logan = ski.data.shepp_logan_phantom()

column_mask = SimpleMask(2).mask_columns(shepp_logan)

map_estimator = MAPEstimator(column_mask, 1e-3, 0.05, 1e-2)

# FFT
#kspace_shepp = np.fft.fft2(shepp_logan)
#k_space = map_estimator.A(shepp_logan)

print(type(column_mask))

plt.figure()
plt.imshow(shepp_logan, cmap='gray')
plt.title('Shepp-Logan phantom')
#print(shepp_logan[134, 201])
#print(np.min(shepp_logan), np.max(shepp_logan), np.mean(shepp_logan))

#plt.figure()
#plt.imshow(masked_shepp_logan, cmap='gray')
#print(simple_column_mask(shepp_logan, 4))

plt.show()
