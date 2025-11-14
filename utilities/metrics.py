import numpy as np

def psnr(x_rec, x_ref, data_range=1.0):
    """Compute PSNR between two magnitude images.
    data_range: 1.0, if image normalized to [0,1]
    
    """
    x_rec = np.abs(x_rec)
    x_ref = np.abs(x_ref)
    max_val = np.max(x_ref)

    # Normalize to the same scale
    x_rec = x_rec / max_val
    x_ref = x_ref / max_val

    mse = np.mean((x_rec - x_ref) ** 2) # (np.mean((np.abs(x_hat) - np.abs(x)) ** 2))
    if mse == 0:
        return np.inf

    return 10 * np.log10((1.0 ** 2/ mse ))

def stats(name, x):
    print(f"{name}: dtype={x.dtype}, shape={x.shape}, max={np.max(np.abs(x)):.2e}, "
          f"min={np.min(np.abs(x)):.2e}, mean={np.mean(np.abs(x)):.2e}, std={np.std(np.abs(x)):.2e}")
    print(f"{name} real: max={np.max(np.real(x)):.2e}, min={np.min(np.real(x)):.2e}")
    print(f"{name} imag: max={np.max(np.imag(x)):.2e}, min={np.min(np.imag(x)):.2e}")
    print("has_nan:", np.isnan(x).any(), "has_inf:", np.isinf(x).any())
    print()