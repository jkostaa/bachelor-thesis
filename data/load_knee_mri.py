from fastmri.data import SliceDataset
from fastmri.data.transforms import ComplexCenterCrop
from torch.utils.data import ConcatDataset

# Load the datasets

train_dataset = SliceDataset(
    root=r"C:\Users\kostanjsek\Documents\knee_mri\knee_singlecoil_train",
    transform=None,
    challenge="singlecoil"
)

val_dataset = SliceDataset(
    root=r"C:\Users\kostanjsek\Documents\knee_mri\knee_singlecoil_val",
    transform=None,
    challenge="singlecoil"
)

test_dataset = SliceDataset(
    root=r"C:\Users\kostanjsek\Documents\knee_mri\knee_singlecoil_test",
    transform=None,
    challenge="singlecoil"
)

# Combined dataset
combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])