"""
Implementations of Torch Dataset and DataLoader that can work with JAX,
although only in a single process (no prefetch) setting.
"""
import numpy as np

from jax.tree_util import tree_map
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import default_collate

from .datasets import Dataset as BaseDataset
from .datasets import PackedDataset as BasePackedDataset

class Dataset(BaseDataset, TorchDataset):
    """
    Same as datasets.Dataset, but inherits torch.utils.data.Dataset and returns NumPy arrays
    with __getitem__
    """
    def __getitem__(self, idx) -> tuple[np.ndarray]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            x, y (np.ndarray): with inputs and targets
        """
        sample_input, sample_target, sample_forcing = self.get_xarrays(idx)

        x = self._stack(sample_input, sample_forcing)
        y = self._stack(sample_target)
        return x.values.squeeze(), y.values.squeeze()


class PackedDataset(BasePackedDataset, TorchDataset):
    def __getitem__(self, idx) -> tuple[np.ndarray]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            x, y (np.ndarray): with inputs and targets
        """
        x = self.inputs["inputs"].isel(sample=idx, drop=True).values
        y = self.targets["targets"].isel(sample=idx, drop=True).values
        return x, y


class DataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = collate_fn
        super().__init__(*args, **kwargs)


def collate_fn(batch):
    return tree_map(np.asarray, default_collate(batch))
