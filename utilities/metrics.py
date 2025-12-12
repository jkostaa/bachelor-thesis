import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ski_ssim

dtype_range = {
    bool: (False, True),
    np.bool_: (False, True),
    float: (-1, 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}

def mean_squared_error(x_rec, x_ref):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    return np.mean((x_rec - x_ref) ** 2, dtype=np.float64)

def psnr(x_rec, x_ref, data_range=None):
    """
    Compute PSNR between two magnitude images.
    data_range: 1.0, if image normalized to [0,1]
    
    """

    if data_range is None:
        dmin, dmax = dtype_range[x_ref.dtype.type]
        true_min, true_max = np.min(x_ref), np.max(x_ref)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range."
            )
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin


    err = mean_squared_error(x_ref, x_rec)
    data_range = float(data_range)  # prevent overflow for small integer types
    return 10 * np.log10((data_range**2) / err)


def stats(name, x):
    print(f"{name}: dtype={x.dtype}, shape={x.shape}, max={np.max(np.abs(x)):.2e}, "
          f"min={np.min(np.abs(x)):.2e}, mean={np.mean(np.abs(x)):.2e}, std={np.std(np.abs(x)):.2e}")
    print(f"{name} real: max={np.max(np.real(x)):.2e}, min={np.min(np.real(x)):.2e}")
    print(f"{name} imag: max={np.max(np.imag(x)):.2e}, min={np.min(np.imag(x)):.2e}")
    print("has_nan:", np.isnan(x).any(), "has_inf:", np.isinf(x).any())
    print()

def nmse(x_rec, x_ref):
    """
    Compute the Normalized Mean Squared Error (NMSE) between a reconstructed image and reference image.
    Parameters:
    - x_rec: reconstructed image, 2D or 3D numpy array
    - x_ref: reference image, same shape as x_recon
    Returns:
    - NMSE value (scalar)
    """
    #     # Ensure the inputs are float
    # x_rec = x_rec.astype(np.float64)
    # x_ref = x_ref.astype(np.float64)
    
    # mse = np.sum(np.abs(x_rec - x_ref)**2)
    # norm = np.sum(np.abs(x_ref)**2)
    
    # return mse / norm

    return mean_squared_error(x_ref, x_rec) * x_ref.size / np.sum(x_ref**2)

def ssim(x_rec, x_ref, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False):
    return ski_ssim(x_rec, x_ref,data_range=1.0)