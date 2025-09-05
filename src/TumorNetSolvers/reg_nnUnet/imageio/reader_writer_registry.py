# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024
import traceback
from typing import Type

from batchgenerators.utilities.file_and_folder_operations import join

import TumorNetSolvers.reg_nnUnet as reg_nnUnet
from TumorNetSolvers.reg_nnUnet.imageio.natural_image_reader_writer import NaturalImage2DIO
from TumorNetSolvers.reg_nnUnet.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from TumorNetSolvers.reg_nnUnet.imageio.simpleitk_reader_writer import SimpleITKIO2
from TumorNetSolvers.reg_nnUnet.imageio.tif_reader_writer import Tiff3DIO
from TumorNetSolvers.reg_nnUnet.imageio.base_reader_writer import BaseReaderWriter
from TumorNetSolvers.reg_nnUnet.utilities.find_class_by_name import recursive_find_python_class

LIST_OF_IO_CLASSES = [
    NaturalImage2DIO,
    SimpleITKIO2,
    Tiff3DIO,
    NibabelIO,
    NibabelIOWithReorient
]


def determine_reader_writer_from_dataset_json(dataset_json_content: dict, example_file: str = None,
                                              allow_nonmatching_filename: bool = False, verbose: bool = True
                                              ) -> Type[BaseReaderWriter]:
    if 'overwrite_image_reader_writer' in dataset_json_content.keys() and \
            dataset_json_content['overwrite_image_reader_writer'] != 'None':
        ioclass_name = dataset_json_content['overwrite_image_reader_writer']
        # trying to find that class in the reg_nnUnet.imageio module
        try:
            ret = recursive_find_reader_writer_by_name(ioclass_name)
            if verbose: print(f'Using {ret} reader/writer')
            return ret
        except RuntimeError:
            if verbose: print(f'Warning: Unable to find ioclass specified in dataset.json: {ioclass_name}')
            if verbose: print('Trying to automatically determine desired class')
    return determine_reader_writer_from_file_ending(dataset_json_content['file_ending'], example_file,
                                                    allow_nonmatching_filename, verbose)


def determine_reader_writer_from_file_ending(file_ending: str, example_file: str = None, allow_nonmatching_filename: bool = False,
                                             verbose: bool = True):
    for rw in LIST_OF_IO_CLASSES:
        if file_ending.lower() in rw.supported_file_endings:
            if example_file is not None:
                # if an example file is provided, try if we can actually read it. If not move on to the next reader
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose: print(f'Using {rw} as reader/writer')
                    return rw
                except:
                    if verbose: print(f'Failed to open file {example_file} with reader {rw}:')
                    traceback.print_exc()
                    pass
            else:
                if verbose: print(f'Using {rw} as reader/writer')
                return rw
        else:
            if allow_nonmatching_filename and example_file is not None:
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose: print(f'Using {rw} as reader/writer')
                    return rw
                except:
                    if verbose: print(f'Failed to open file {example_file} with reader {rw}:')
                    if verbose: traceback.print_exc()
                    pass
    raise RuntimeError(f"Unable to determine a reader for file ending {file_ending} and file {example_file} (file None means no file provided).")


def recursive_find_reader_writer_by_name(rw_class_name: str) -> Type[BaseReaderWriter]:
    ret = recursive_find_python_class(join(reg_nnUnet.__path__[0], "imageio"), rw_class_name, 'reg_nnUnet.imageio')
    if ret is None:
        raise RuntimeError("Unable to find reader writer class '%s'. Please make sure this class is located in the "
                           "reg_nnUnet.imageio module." % rw_class_name)
    else:
        return ret
