# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from TumorNetSolvers.reg_nnUnet.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from TumorNetSolvers.reg_nnUnet.training.dataloading.nnunet_dataset import nnUNetDataset

import time 
class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        start_time = time.time()
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        pred_all = np.zeros(self.seg_shape, dtype=np.float32)
        mask_all = np.zeros(self.data_shape, dtype=np.float32)  # Define mask storage
        case_properties = []
        params = []

        for j, i in enumerate(selected_keys):
            data, mask, param, pred, properties = self._data.load_case(i)
            print(param)
            print(mask)
            params.append(param)
            case_properties.append(properties)

            # Define slicing and padding for each case
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=properties.get('class_locations', None))

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            
            # Apply slicing
            data = data[this_slice]
            pred = pred[this_slice]

            # Define padding for current shape
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            
            # Apply padding
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            pred_all[j] = np.pad(pred, padding, 'constant', constant_values=0)
            mask_all[j] = np.pad(mask, padding, 'constant', constant_values=0)  # Store padded mask

        # Apply transformations if specified
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    pred_all = torch.from_numpy(pred_all).float()
                    mask_all = torch.from_numpy(mask_all).float()  # Convert mask to torch tensor
                    
                    images, preds, masks = [], [], []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': pred_all[b], 'mask': mask_all[b]})
                        images.append(tmp['image'])
                        preds.append(tmp['segmentation'])
                        masks.append(tmp['mask'])
                    
                    # Stack transformed images, preds, and masks
                    data_all = torch.stack(images)
                    mask_all = torch.stack(masks)
                    
                    if isinstance(preds[0], list):
                        pred_all = [torch.stack([s[i] for s in preds]) for i in range(len(preds[0]))]
                    else:
                        pred_all = torch.stack(preds)

        return {'data': data_all, 'masks': mask_all, 'params': torch.stack(params), 'target': pred_all, 'keys': selected_keys}



if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
