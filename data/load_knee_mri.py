import os
from fastmri.data import SliceDataset

# from fastmri.data.transforms import ComplexCenterCrop
# from torch.utils.data import ConcatDataset


def load_fastmri_data(base_path=None):
    if base_path is None:
        base_path = os.environ.get("FASTMRI_DATA_PATH")

    if base_path is None:
        raise RuntimeError("Please set the FASTMRI_DATA_PATH environment variable.")

    paths = {
        "train": os.path.join(base_path, "knee_singlecoil_train", "singlecoil_train"),
        "val": os.path.join(base_path, "knee_singlecoil_val", "singlecoil_val"),
        "test": os.path.join(base_path, "knee_singlecoil_test", "singlecoil_test"),
    }

    datasets = {}

    for key, path in paths.items():
        try:
            datasets[key] = SliceDataset(
                root=path, transform=None, challenge="singlecoil"
            )
        except Exception as e:
            print(f"Failed to load {key}_dataset:", e)
            datasets[key] = None

    return datasets["train"], datasets["val"], datasets["test"]


if __name__ == "__main__":
    os.environ["FASTMRI_DATA_PATH"] = r"C:\Users\kostanjsek\Documents\knee_mri"
    train_dataset, val_dataset, test_dataset = load_fastmri_data()

# Sanity check
# print("train_dataset loaded:", isinstance(train_dataset, SliceDataset))
# print("val_dataset loaded:", isinstance(val_dataset, SliceDataset))
# print("test_dataset loaded:", isinstance(test_dataset, SliceDataset))

# Combined dataset
# combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
