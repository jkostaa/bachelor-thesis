import numpy as np

def psnr(x_rec, x_ref, data_range=1.0):
    """Compute PSNR between two magnitude images.
    data_range: 1.0, if image normalized to [0,1]
    
    """
    x_rec = np.abs(x_rec)
    x_ref = np.abs(x_ref)
    
    # Normalize to the same scale
    x_rec = x_rec / np.max(x_ref)
    x_ref = x_ref / np.max(x_ref)
    
    mse = np.mean((x_rec - x_ref) ** 2) # (np.mean((np.abs(x_hat) - np.abs(x)) ** 2))
    if mse == 0:
        return np.inf
    max_val = np.max(x_ref)
    return 10 * np.log10((max_val ** 2/ np.sqrt(mse)))