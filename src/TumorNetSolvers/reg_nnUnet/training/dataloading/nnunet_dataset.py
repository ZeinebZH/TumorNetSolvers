# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024
import os
from typing import List
import numpy as np
import shutil
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from TumorNetSolvers.reg_nnUnet.training.dataloading.utils import get_case_identifiers

class nnUNetDataset:
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 param_file: str = '',
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        Initialize a dictionary where keys are training case names and values contain relevant information per case.
        """
        super().__init__()
        # Set identifiers
        case_identifiers = case_identifiers or get_case_identifiers(folder)
        case_identifiers.sort()

        # Load parameter dictionary if provided
        self.param_dict = torch.load(param_file) if param_file else {}
        
        # Set up the dataset dictionary with paths and loaded params
        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {
                'data_file': join(folder, f"{c}.npz"),
                'properties_file': join(folder, f"{c}.pkl"),
                'params': self.param_dict.get(c)
            }
            if folder_with_segs_from_previous_stage:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        # Load properties into RAM for small datasets
        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for case in self.dataset.keys():
                self.dataset[case]['properties'] = load_pickle(self.dataset[case]['properties_file'])

        # Determine if files should be kept open
        self.keep_files_open = os.getenv('nnUNet_keep_files_open', '0').lower() in ('true', '1', 't')

    def __getitem__(self, key):
        item = {**self.dataset[key]}
        if 'properties' not in item:
            item['properties'] = load_pickle(item['properties_file'])
        return item

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def load_case(self, key):
        entry = self[key]
        data, seg, mask = None, None, None

        # Load data with fallback to npy file if necessary
        if 'open_data_file' in entry:
            data = entry['open_data_file']
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = np.load(entry['data_file'])['data']

        # Load segmentation with fallback to npy file if necessary
        if 'open_seg_file' in entry:
            seg = entry['open_seg_file']
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
        else:
            seg = np.load(entry['data_file'])['seg']

        # Load additional segmentation from a previous stage if available
        if 'seg_from_prev_stage_file' in entry:
            seg_prev_path = entry['seg_from_prev_stage_file'][:-4] + ".npy"
            seg_prev = np.load(seg_prev_path, 'r') if isfile(seg_prev_path) else np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        # Load mask with fallback to npy file if necessary
        if 'open_mask_file' in entry:
            mask = entry['open_mask_file']
        elif isfile(entry['data_file'][:-4] + "_mask.npy"):
            mask = np.load(entry['data_file'][:-4] + "_mask.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_mask_file'] = mask
        else:
            mask = np.load(entry['data_file'])['mask']

        return data, mask, entry['params'], seg, entry['properties']


    # Define helper methods for dictionary-like behavior
    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return len(self.dataset)

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()
