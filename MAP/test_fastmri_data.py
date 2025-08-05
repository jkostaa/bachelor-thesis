import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
#from fastmri.data import SliceDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load_knee_mri import load_fastmri_data
from map_tv_minimize import MAPEstimator
from masks import BernoulliMask, CenteredBernoulliMask, VariableDensityMask

train_dataset, val_dataset, test_dataset = load_fastmri_data(r"C:\Users\kostanjsek\Documents\knee_mri")

#print("Train dataset loaded:", isinstance(train_dataset, object))
#print("Val dataset length:", len(val_dataset))

def load_sample(split='train', index=0):
    """
    Load a sample from the specified dataset split.
    """

    if split == 'train':
        dataset = train_dataset
    elif split == 'val':
        dataset = val_dataset
    elif split == 'test':
        dataset = test_dataset
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
    
    sample = dataset[index]
    kspace, target = sample[0], sample[1]
    return kspace, target

def zero_filled_ifft(kspace):
    '''
    Apply inverse FFT, and normalize to [0,1] for the visualization
    '''
    reconstructed = np.fft.ifft2(kspace)
    return np.abs(reconstructed) / np.max(np.abs(reconstructed))

def apply_mask(kspace, mask_type='variable_density', seed = 42, **kwargs):
    shape = kspace.shape

    if mask_type == 'bernoulli':
        mask = BernoulliMask(kwargs.get('rate', 0.85), seed=seed).generate(shape)
    elif mask_type == 'centered_bernoulli':
        mask = CenteredBernoulliMask(kwargs.get('center_prob', 0.75), kwargs.get('edge_prob', 0.25), seed=seed).generate(shape)
    elif mask_type == 'variable_density':
        mask = VariableDensityMask(kwargs.get('type', 'gaussian'), kwargs.get('decay', 2), seed=seed).generate(shape)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    y = kspace * mask
    return mask, y

def map_reconstruction(y, mask, **map_params):
    map_estimator = MAPEstimator(mask, **map_params) 
    return map_estimator.subgradient_descent(y)

def plot_graphs(original, masked_ifft, map_reconstruct, target=None):
    
    plt.subplot(1, 3, 1)
    plt.title("Zero-filled iFFT")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("iFFT of Masked k-space")
    plt.imshow(masked_ifft, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("MAP Reconstruction")
    plt.imshow(map_reconstruct, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if target is not None:
        plt.figure()
        plt.title("Ground Truth")
        plt.imshow(target, cmap='gray')
        plt.axis('off')
        plt.show()

def run_pipeline(split='train', index=0, mask_type='bernoulli', map_params=None, mask_params=None):
    map_params = map_params or dict(sigma=0.95, lambda_=0.01, eps=1e-2, learning_rate=0.01, max_iter=100)
    mask_params = mask_params or dict(rate=0.85)
    
    kspace, target = load_sample(split=split, index=index)
    zero_filled = zero_filled_ifft(kspace)

    mask, y = apply_mask(kspace, mask_type=mask_type, **mask_params)
    map_reconstruct = map_reconstruction(y, mask, **map_params)
    masked_ifft = np.abs(np.fft.ifft2(y))

    plot_graphs(zero_filled, masked_ifft, map_reconstruct, target=target)



def main():
    run_pipeline(
        split='val',
        index=123,
        mask_type='variable_density',
        map_params={'sigma': 0.95, 'lambda_': 0.01, 'eps': 1e-2, 'learning_rate': 0.01, 'max_iters': 100},
        mask_params={'rate': 0.85}
    )
    

if __name__ == "__main__":
    main()
    