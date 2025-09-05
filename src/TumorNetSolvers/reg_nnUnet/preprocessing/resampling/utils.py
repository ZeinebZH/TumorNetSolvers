# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024
from typing import Callable

import TumorNetSolvers.reg_nnUnet as reg_nnUnet
from batchgenerators.utilities.file_and_folder_operations import join
from TumorNetSolvers.reg_nnUnet.utilities.find_class_by_name import recursive_find_python_class


def recursive_find_resampling_fn_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(join(reg_nnUnet.__path__[0], "preprocessing", "resampling"), resampling_fn,
                                      'reg_nnUnet.preprocessing.resampling')
    if ret is None:
        raise RuntimeError("Unable to find resampling function named '%s'. Please make sure this fn is located in the "
                           "reg_nnUnet.preprocessing.resampling module." % resampling_fn)
    else:
        return ret
