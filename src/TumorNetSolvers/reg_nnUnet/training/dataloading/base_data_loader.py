# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024

from typing import Union, Tuple
from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from TumorNetSolvers.reg_nnUnet.training.dataloading.nnunet_dataset import nnUNetDataset

class nnUNetDataLoaderBase(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 transforms=None):
        super().__init__(data, batch_size, 1, None, True, False, True, None)
        self.indices = list(data.keys())

        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.transforms = transforms

    def determine_shapes(self):
        # load one case
        data, _,_, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_color_channels, *self.patch_size)  # Ensure matching shapes for regression
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool = False, class_locations: Union[dict, None] = None,
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        """
        Determines the bounding box for sampling patches from the image.

        Parameters:
        - data_shape: Shape of the data (excluding batch size).
        - force_fg: Whether to enforce that the bounding box should contain foreground regions (not applicable for regression tasks).
        - class_locations: Dictionary of class locations (not used in regression tasks).
        - overwrite_class: Specific class to overwrite (not used in regression tasks).
        - verbose: Whether to print verbose messages.

        Returns:
        - bbox_lbs: Lower bounds of the bounding box.
        - bbox_ubs: Upper bounds of the bounding box.
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        # Compute padding required to ensure the patch size fits within the data shape
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # Define lower and upper bounds for bounding box sampling
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # Sample random bounding box within the defined bounds
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        #print(f"Data shape: {data_shape}, Patch size: {self.patch_size}")
        #print(f"Sampled bbox - Lower bounds: {bbox_lbs}, Upper bounds: {bbox_ubs}")
    
        return bbox_lbs, bbox_ubs
