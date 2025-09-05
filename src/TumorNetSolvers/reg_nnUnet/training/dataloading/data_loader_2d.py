# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from TumorNetSolvers.reg_nnUnet.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from TumorNetSolvers.reg_nnUnet.training.dataloading.nnunet_dataset import nnUNetDataset

class nnUNetDataLoader2D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # Preallocate memory for data and segmentation
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)  # Changed to float32 for regression
        case_properties = []

        for j, current_key in enumerate(selected_keys):
            # Load data and segmentation
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            # Determine whether to oversample foreground (not applicable for regression, hence False)
            force_fg = False

            # Select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key if (
                            len(properties['class_locations'][self.annotated_classes_key]) > 0) else None
                else:
                    selected_class_or_region = None
            else:
                # Filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # If annotated_classes_key locations are present and other classes are present, remove the annotated_classes_key from the list
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None

            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]

            # Prepare class_locations for bbox calculation
            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else False,
                                               class_locations, overwrite_class=selected_class_or_region)

            # Crop the data to the region of the bbox that lies within the data
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # Apply padding
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)  # Changed to 0 for regression

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).float()  # Changed to float for regression
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


