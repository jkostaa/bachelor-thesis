"""
This file is for running the MAP estimation on skimage.data:
    - Load images from skimage
    - Generates k-space with FFT
    - Simulate undersampling
    - Runs the reconstruction
    - Display/compare results
"""

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

# from skimage.transform import rescale, resize

from map_tv_minimize import MAPEstimator
from utilities.masks import SimpleMask, BernoulliMask, CenteredBernoulliMask, VariableDensityMask

# np.set_printoptions(threshold=np.inf)


# load the image
def load_image():
    return ski.data.shepp_logan_phantom()


# create mask (in the F domain of course)
def create_mask(shepp_logan):
    masks = {
        "column": SimpleMask(2).mask_columns(np.zeros(shepp_logan)),
        "row": SimpleMask(3).mask_rows(np.zeros(shepp_logan)),
        "bernoulli": BernoulliMask(0.88, seed=30).generate(shepp_logan),
        "centered_bernoulli": CenteredBernoulliMask(0.75, 0.25, seed=30).generate(
            shepp_logan
        ),
        "variable_density": VariableDensityMask("gaussian", 2, seed=30).generate(
            shepp_logan
        ),
    }
    return masks


# the 'actual' measurement
def reconstruct_image(y, mask, shape):
    # y = vd_mask * np.fft.fft2(shepp_logan)
    map_estimator = MAPEstimator(mask, 0.95, 0.01, 1e-2, 0.1, 100)
    return map_estimator.subgradient_descent(y), map_estimator


# Langevin
def run_langevin(map_estimator, y):
    samples = map_estimator.langevin_sampling(y, 150, 20, 3)
    posterior_mean = np.mean(
        samples, axis=0
    )  # Compute a point estimate (e.g., posterior mean)
    posterior_std = np.std(
        samples, axis=0
    )  # Estimate uncertainty (e.g., pixel-wise variance)
    return posterior_mean, posterior_std


def show_plots(original, reconstructed, y, posterior_mean):
    """
    Plotting the results
    """

    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(original, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed image")
    plt.imshow(np.abs(reconstructed), cmap="gray")
    plt.subplot(1, 3, 3)

    plt.title("ifft of y")
    plt.imshow(np.abs(np.fft.ifft2(y)), cmap="gray")

    # plt.subplot(1, 4, 4)
    # plt.title("x_init")
    # plt.imshow(np.random.rand(400,400), cmap='gray')

    plt.show()

    # Posterior mean
    plt.imshow(posterior_mean, cmap="gray")
    plt.title("Posterior Mean Reconstruction")
    plt.colorbar()
    plt.show()


def main():
    image = load_image()
    masks = create_mask(image.shape)

    # Choose the mask you want to test with
    mask = masks["variable_density"]

    # Simulated measurement
    y = mask * np.fft.fft2(image)

    # Reconstruct
    img_reconstruct, map_estimator = reconstruct_image(y, mask, image.shape)

    # Posterior estimation
    posterior_mean, posterior_std = run_langevin(map_estimator, y)

    # Show results
    show_plots(image, img_reconstruct, y, posterior_mean)


if __name__ == "__main__":
    main()
