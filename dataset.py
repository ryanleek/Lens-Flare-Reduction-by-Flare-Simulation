#CUDA_LAUNCH_BLOCKING = "1"

import torch
from skimage.io import imread
from torch.utils import data
from tqdm import tqdm


class SegmentationDataSet(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        transform=None,
        use_cache=False,
        pre_transform=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool() as pool:
                self.cached_data = pool.starmap(
                    self.read_images, zip(inputs, targets, repeat(self.pre_transform))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(str(inp)), imread(str(tar))
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar