# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Major modifications by Zeineb Haouari
import inspect
import os
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List, Literal

import wandb
import numpy as np
import torch
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from torch._dynamo import OptimizedModule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from TumorNetSolvers.utils.metrics import EMA, compute_dice_score, compute_ssim
from TumorNetSolvers.utils.train_val_split import train_val_test_split_size, train_val_test_split_ratio, train_val_test_split_fx
from TumorNetSolvers.preprocessing.target_handling import RegressionManager
from TumorNetSolvers.models.dynamic_Unets import get_network_from_plans_new
from TumorNetSolvers.models.ViT import CombinedVisionTransformer3D
from TumorNetSolvers.models.embeddings import PatchEmbed3D
from TumorNetSolvers.models.tumor_surrogate_net import TumorSurrogate


from TumorNetSolvers.reg_nnUnet.training.loss.deep_supervision import DeepSupervisionWrapper
from TumorNetSolvers.reg_nnUnet.training.lr_scheduler.polylr import PolyLRScheduler
from TumorNetSolvers.reg_nnUnet.training.dataloading.nnunet_dataset import nnUNetDataset
from TumorNetSolvers.reg_nnUnet.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from TumorNetSolvers.reg_nnUnet.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from TumorNetSolvers.reg_nnUnet.training.data_augmentation.reg_transforms import BasicTransform2, SpatialTransform2
from TumorNetSolvers.reg_nnUnet.training.dataloading.utils import get_case_identifiers, unpack_dataset
from TumorNetSolvers.reg_nnUnet.training.data_augmentation.compute_initial_patch_size import get_patch_size
from TumorNetSolvers.reg_nnUnet.training.logging.nnunet_logger import nnUNetLogger
from TumorNetSolvers.reg_nnUnet.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from TumorNetSolvers.reg_nnUnet.utilities.collate_outputs import collate_outputs
from TumorNetSolvers.reg_nnUnet.utilities.helpers import empty_cache, dummy_context
from TumorNetSolvers.reg_nnUnet.utilities.plans_handling.plans_handler import PlansManager
from TumorNetSolvers.reg_nnUnet.configuration import ANISO_THRESHOLD


from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, isfile, load_json
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from TumorNetSolvers.utils.paths import set_environment_variables
set_environment_variables()
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

# If they are not set, raise an error or handle the missing paths
if not nnUNet_preprocessed or not nnUNet_results:
    raise EnvironmentError("One or more environment variables (nnUNet_preprocessed, nnUNet_results) are not set.")

def init_weights(layer):
    """Initializes weights for Linear layers using Kaiming He initialization and biases to zero."""
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # He initialization for ReLU
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class Trainer(object):
    def __init__(self, plans: dict, configuration: str, dataset_json: dict, signature:str, fold: Union[Literal['train_val', 'all'], int]= "train_val",  unpack_dataset: bool = True, loss_func: Literal['mse', 'mae']= 'mse', model: Literal['nnUnet', 'ViT', 'TumorSurrogate']= "ViT",
                 device: torch.device = torch.device(f'cuda:5'), project_name="NN-based-tumor-solvers"):
        self.signature=signature #Must Be Defined!!
        self.enable_deep_supervision = False
        self.model= model
        self.gpu_id=  int(device.index)  
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = self.gpu_id if not self.is_ddp else dist.get_rank()
        self.device = device
        self.loss_fn= loss_func
        self.mask= nn.ReLU()

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            print("Rank",self.rank)
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=self.gpu_id)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)

        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset
        self.global_step = 0
        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}',  f'_{self.signature}_{self.model}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)

        self.param_file = join(self.preprocessed_dataset_folder_base,"param_dict.pth")
        print(self.param_file)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.num_iterations_per_epoch =  400 #ideally dataset_sz//batch_sz
        self.num_val_iterations_per_epoch = 100
        self.num_epochs = 1000
        self.current_epoch = 0

        self.num_input_channels = None  # -> self.initialize()
        
        if self.model=="ViT":
            #####TODO: max_volume_size should be extracted automatically (dataset_fingerprint.json file)
            self.network = CombinedVisionTransformer3D(max_volume_size=self.configuration_manager.patch_size[0], patch_size=16, in_chans=1, num_classes=1000, embed_dim=384, depth=12,
                    num_heads=6, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D, norm_layer=None,
                    act_layer=None, weight_init='', global_pool=False, param_dim=5).to(self.device)
        
        elif self.model=="TumorSurrogate":
            self.network = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1]).to(self.device)
            self.network.apply(init_weights)
        elif self.model=="nnUnet":
             self.enable_deep_supervision = True


        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        self._best_ema_dice = None
        self._best_val_dice= None
        self._best_ema_loss = None
        self._best_val_loss= None
        self.dice_ema = EMA(alpha=0.1)
        self.loss_ema = EMA(alpha=0.1)
        ### inference things
        self.inference_allowed_mirroring_axes = None  # ->self.configure_rotation_dummyDA_mirroring_and_inital_patch_size  (will be saved in checkpoints)

        ### checkpoint saving 
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size()
        self.project_name=project_name #"NN-based-tumor-solvers"

        wandb.init(project=self.project_name, settings=wandb.Settings(_disable_stats=True))
        
        self.was_initialized = False
        


    def initialize(self):
        if not self.was_initialized:
            
            if self.model=="nnUnet":
                self.num_input_channels = RegressionManager.determine_num_input_channels(self.dataset_json)
                self.network = self.build_network_architecture(
                    "PlainConvUnetNew",
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            self.loss = self._build_loss()

            self.was_initialized = True
            self.gradients = {}
            self.register_hooks()
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        
    def _do_i_compile(self):
        # new default: compile is enabled!

        # compile does not work on mps
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False

        # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        if os.name == 'nt':
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                       "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
            return False

        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
        
    
    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int = 1,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        return get_network_from_plans_new(
            arch_class_name=architecture_class_name,
            arch_kwargs=arch_init_kwargs,
            arch_kwargs_req_import=arch_init_kwargs_req_import,
            input_channels=num_input_channels,
            output_channels=num_output_channels,
            inputs_shape=self.shape_data,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
    

    def _set_batch_size(self):
        if not self.is_ddp:
          self.batch_size = self.configuration_manager.batch_size
        else:
          # DDP: distribute the batch size across GPUs

          world_size = dist.get_world_size()
          my_rank = dist.get_rank()

          global_batch_size = self.configuration_manager.batch_size
          assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of GPUs.'

          # Calculate batch size per GPU
          batch_size_per_GPU = [global_batch_size // world_size] * world_size
          batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                else batch_size_per_GPU[i]
                                for i in range(len(batch_size_per_GPU))]
          assert sum(batch_size_per_GPU) == global_batch_size

          # Set batch size for this GPU
          self.batch_size = batch_size_per_GPU[my_rank]

          print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])
    

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
    

    def _build_loss(self):
        if self.loss_fn== "mae":
            loss =nn.L1Loss() # MAE
        else:
            loss =nn.MSELoss() # MSE 
            
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Configures rotation, dummy 2D data augmentation, and initial patch size for a regression task.
        The goal is to adapt these parameters for cases where tumors appear in different regions with varying shapes.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # Adjust rotation angles for regression tasks, allowing small rotations to handle variations in tumor location and shape
        if dim == 2:
            do_dummy_2d_data_aug = False
            rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # Slightly restrict the rotation range for 2D slices in 3D volumes
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError("Unsupported dimensionality: expected 2D or 3D.")

        # Recalculate initial patch size based on the adapted rotation and scaling parameters
        initial_patch_size = get_patch_size(
            patch_size[-dim:],  # Focus on the last `dim` dimensions
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.9, 1.1)  # Adjusted scaling to keep augmentations moderate for regression tasks
        )

        # Handle the case where 2D augmentation is applied in a 3D context
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]  # Maintain the original depth if augmenting in 2D slices

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == self.gpu_id:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)


    def configure_optimizers(self):
            """
            Configure the optimizer and learning rate scheduler for training.

            This function checks if 'self.network' is a function. If it is, it raises a TypeError to provide a more informative
            error message, indicating that a PyTorch module (nn.Module) was expected. Otherwise, it proceeds with configuring
            the optimizer and learning rate scheduler.
            """
            if self.model=="ViT":
                optimizer = torch.optim.AdamW(self.network.parameters() , lr=2e-4, weight_decay=1e-2)
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=0.0008,  # Larger initial LR, suited for exploration
                    total_steps=self.num_epochs, 
                    pct_start=0.1,  # 10% of epochs for gradual ramp-up
                    anneal_strategy='linear',
                    cycle_momentum=False,
                    div_factor=25,  # Balancing the initial LR
                    final_div_factor=10  # Moderate final LR to avoid overshooting
                )
            else:                   
            #elif self.model=="TumorSurrogate":
                optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-12)
                lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

                #scheduler = torch.optim.lr_scheduler.OneCycleLR(
                #    optimizer, 
                #    max_lr=0.0008, 
                #    epochs=self.num_epochs, 
                #    steps_per_epoch= self.num_iterations_per_epoch
                #)


            #else:
                #optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                            #momentum=0.99, nesterov=True)
                #lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
            return optimizer, lr_scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.decoder.deep_supervision = enabled

    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return

        if self.local_rank == self.gpu_id:
            try:

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)
            finally:
                empty_cache(self.device)

                
    def do_split(self):
        """
        Splits the dataset into training and validation sets based on the specified fold.
        Supports both 'all' and 'train_val' configurations.
        """
        # Initialize keys for training and validation
        tr_keys = []
        val_keys = []

        # Load dataset and define the splits file path
        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                param_file=self.param_file, num_images_properties_loading_threshold=0)

        if self.fold == "train_val":
            all_keys_sorted = sorted(dataset.keys())
            print(splits_file)
            splits_f=load_json(splits_file)
            print(splits_f)
            val_keys=splits_f[0]["val"]
            test_keys=splits_f[0]["test"]
            tr_keys, val_keys, test_keys = train_val_test_split_fx(all_keys_sorted,train_size=10000, val_size=2000, test_size=2000,  fixed_val_test=(val_keys, test_keys))
            splits =[]
            splits.append({})
            splits[-1]['train'] = list(tr_keys)
            splits[-1]['val'] = list(val_keys)
            splits[-1]['test'] = list(test_keys)
            save_json(splits, splits_file)
            self.print_to_log_file(f"Train-Validation-Test Split: {len(tr_keys)} training cases, {len(val_keys)} validation cases, and  {len(test_keys)} test cases .")
        
        return tr_keys, val_keys, test_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys, _ = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   param_file= self.param_file,
                            
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    param_file= self.param_file,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val


    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # Determine deep supervision scales (optional, based on regression needs)
        deep_supervision_scales = self._get_deep_supervision_scales()

        # Configure data augmentation and initial patch size
        (   rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Training pipeline transformations for regression
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            self.configuration_manager.use_mask_for_norm,
        )
        
        # Validation pipeline transformations for regression
        val_transforms = self.get_validation_transforms(deep_supervision_scales)

        # Load datasets for training and validation
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        wandb.config.update({'Training Set Size':len(dataset_tr), 
                             'Validation Set Size':len(dataset_val) })
        if dim == 2:
            # 2D case
            dl_tr = nnUNetDataLoader2D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,  # Use adjusted initial patch size for training
                self.configuration_manager.patch_size,  # Use final patch size for training
                transforms=tr_transforms  # Apply training transformations
            )

            dl_val = nnUNetDataLoader2D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,  # Use final patch size for validation
                self.configuration_manager.patch_size,  # Same patch size for validation
                transforms=val_transforms  # Apply validation transformations
            )

        else:
            # 3D case
            dl_tr = nnUNetDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,  # Use adjusted initial patch size for training
                self.configuration_manager.patch_size,  # Use final patch size for training
                transforms=tr_transforms  # Apply training transformations
            )

            dl_val = nnUNetDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,  # Use final patch size for validation
                self.configuration_manager.patch_size,  # Same patch size for validation
                transforms=val_transforms  # Apply validation transformations
            )

        # Determine number of allowed processes for data augmentation
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                    transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                    num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                    pin_memory=self.device.type == 'cuda',
                                                    wait_time=0.002)
        # Initialize the data generators
        init_mt_gen_train = next(mt_gen_train)
        _ = next(mt_gen_val)
        # Obtain the shape of data
        shape_data = init_mt_gen_train['data'].shape # Both are the same
        return mt_gen_train, mt_gen_val, shape_data


    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            ) -> BasicTransform2:
        transforms = []
        
        patch_size_spatial = patch_size
        ignore_axes = None
        transforms.append(
            SpatialTransform2(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

              
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))


        return ComposeTransforms(transforms)


    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None]) -> BasicTransform2:
        transforms = []
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)


    def on_train_start(self):
        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present when doing inference
        self.dataloader_train, self.dataloader_val, self.shape_data = self.get_dataloaders()

        if not self.was_initialized:
            self.initialize()


        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        print(self.enable_deep_supervision)
        if self.model=="nnUnet": 
            self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)
        # maybe unpack
        if self.unpack_dataset and self.local_rank == self.gpu_id:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
            self.print_to_log_file('unpacking done...')
        if self.is_ddp:
            dist.barrier()
        """
        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))
        """
        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()
    

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == self.gpu_id and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        param=batch['params']
        
        data = data.to(self.device, non_blocking=True)
        param = param.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [self.mask(i).to(self.device, non_blocking=True) for i in target]
        else:
            target = self.mask(target).to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, param)
            # del data
            l = self.loss(output, target)
            print('loss:', l)
            predicted = output
            if self.enable_deep_supervision:
                predicted=predicted[0]
                target=target[0]
            print("pred shape: ", predicted.shape, f"max_v {predicted.max()}, min_v {predicted.min()} "
              "\n target shape: ", target.shape, f"max_v {target.max()}, min_v {target.min()})")
            d = compute_dice_score(u_pred=predicted, u_sim=target, threshold=0.4)
            
            ssim= compute_ssim(predicted ,target)
            print(d, ssim)
            print(l)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            self.lr_scheduler.step()
            
        wandb.log({
            f'train_{self.loss_fn}': l.item(),
            'train_dice': d.item(), 'train_ssim': ssim.item() }, step=self.global_step)
        self.global_step+=1    
        
        return {'loss': l.detach().cpu().numpy(), 'dice': d.detach(), 'SSIM': ssim.detach()}
    
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dice_scores_tr = [None for _ in range(dist.get_world_size())]
            ssim_scores_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(dice_scores_tr, outputs['dice'])
            dist.all_gather_object(ssim_scores_tr, outputs['SSIM'])
            loss_here = np.vstack(losses_tr).mean()
            dice_here = np.vstack(dice_scores_tr).mean()
            ssim_here = np.vstack(ssim_scores_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            dice_here = torch.mean(outputs['dice'])
            ssim_here = torch.mean(outputs['SSIM'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_dice', dice_here, self.current_epoch)
        print("train_ssim", ssim_here)
        self.logger.log('train_ssim', ssim_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        param = batch['params']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [self.mask(i).to(self.device, non_blocking=True) for i in target]
        else:
            target = self.mask(target).to(self.device, non_blocking=True)
        param = param.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, param)
            del data, param
            l = self.loss(output, target)

        print("Deep supervision?", self.enable_deep_supervision)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
            print(output.shape, target.shape)

        predicted = output
        print("pred shape: ", predicted.shape, f"max_v {predicted.max()}, min_v {predicted.min()} "
              "\n target shape: ", target.shape, f"max_v {target.max()}, min_v {target.min()})")
        
        dice = compute_dice_score(u_pred=predicted, u_sim=target, threshold=0.4)
        print('dice',dice)
        ssim = compute_ssim(predicted, target)
        print("SSIM:", ssim.item())
        

        print(f'val_loss: {l.detach().cpu().numpy()}, val_dice: {dice}, val_ssim:{ssim.item()}')
        return {'loss': l.detach().cpu().numpy(), 'dice': dice, 'SSIM': ssim.item()}
        

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dices_val = [None for _ in range(world_size)]
            ssim_val = [None for _ in range(world_size)]

            dist.all_gather_object(losses_val, outputs_collated['loss'])
            dist.all_gather_object(dices_val, outputs_collated['dice'])
            dist.all_gather_object(ssim_val, outputs_collated['SSIM'])
            print(dices_val)
            print(np.vstack(dices_val))
            dice_here = np.vstack(dices_val).mean()
            loss_here = np.vstack(losses_val).mean()
            ssim_here = np.vstack(ssim_val).mean()
        else:
            dice_here = torch.mean(outputs_collated['dice'])
            loss_here = np.mean(outputs_collated['loss'])
            ssim_here = np.mean(outputs_collated['SSIM'])

        # Update EMA
        ema_dice = self.dice_ema.update(dice_here)
        ema_loss= self.loss_ema.update(loss_here)

        self.logger.log('val_dice', dice_here, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_ssim', ssim_here, self.current_epoch)
        self.logger.log('ema_dice', ema_dice, self.current_epoch)  # Log EMA dice
        self.logger.log('ema_loss', ema_loss, self.current_epoch)  
        wandb.log({
            f'val_{self.loss_fn}': loss_here,
            'val_dice': dice_here,
            'val_ssim': ssim_here,
            'ema_dice': ema_dice,  # Log EMA dice to wandb
            'ema_loss' : ema_loss
        }, step=self.global_step)
            

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
    
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_dice', np.round(self.logger.my_fantastic_logging['val_dice'][-1].cpu(), decimals=4))
        self.print_to_log_file('val_ssim', np.round(self.logger.my_fantastic_logging['val_ssim'][-1], decimals=4))
        self.print_to_log_file('ema_dice', np.round(self.logger.my_fantastic_logging['ema_dice'][-1].cpu(), decimals=4))
        self.print_to_log_file('ema_loss', np.round(self.logger.my_fantastic_logging['ema_loss'][-1], decimals=4))
        
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing using EMA dice
        if self._best_ema_dice is None or self.logger.my_fantastic_logging['ema_dice'][-1] > self._best_ema_dice:
            self._best_ema_dice = self.logger.my_fantastic_logging['ema_dice'][-1]
            self.print_to_log_file(f"New best EMA Dice: {np.round(self._best_ema_dice.cpu(), decimals=4)}")
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{self.model}_best_ema_dice.pth'))

        # handle 'best' checkpointing using val dice
        if self._best_val_dice is None or self.logger.my_fantastic_logging['val_dice'][-1] > self._best_val_dice:
            self._best_val_dice = self.logger.my_fantastic_logging['val_dice'][-1]
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{self.model}_best_val_dice.pth'))
        
        if self._best_ema_loss is None or self.logger.my_fantastic_logging['ema_loss'][-1] < self._best_ema_loss:
            self._best_ema_loss = self.logger.my_fantastic_logging['ema_loss'][-1]
            self.print_to_log_file(f"New best EMA loss: {np.round(self._best_ema_loss, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{self.model}_best_ema_loss.pth'))

        # handle 'best' checkpointing using val loss
        if self._best_val_loss is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_val_loss:
            self._best_val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{self.model}_best_val_loss.pth'))
        

        if self.local_rank == self.gpu_id:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == self.gpu_id:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema_dice': self._best_ema_dice,  
                    '_best_val_dice':  self._best_val_dice,  
                    '_best_ema_loss': self._best_ema_loss,  
                    '_best_val_loss':  self._best_val_loss,  
                    'dice_ema_value': self.dice_ema.get_value(), 
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()
            wandb.config.update({
            'Optimizer': self.optimizer,
            'lr_scheduler' :self.lr_scheduler
            })

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_val_dice = checkpoint['_best_val_dice'] 
        self._best_val_loss=checkpoint['_best_val_loss'] 
        self._best_ema_dice = checkpoint['_best_ema_loss'] 
        try:
            self._best_ema_dice = checkpoint['_best_ema_dice'] 
            self.dice_ema.value = checkpoint['dice_ema_value']  
        except KeyError:
            pass
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
    
    def register_hooks(self):
        """Register hooks to capture gradients."""
        for name, layer in self.network.named_children():
            layer.register_backward_hook(self.save_gradient(name))

    def save_gradient(self, name):
        """Hook function to save gradients."""
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]  # Store the gradient output
        return hook
    
    def run_training(self):
        self.on_train_start()
        
        # log important settings
        wandb.config.update({
        'Model': self.model,
        'loss_fn': self.loss_fn,
        'Deep Supervision': self.enable_deep_supervision,
        'initial_lr': self.initial_lr,
        'weight_decay' :self.weight_decay,
        'num_iters_per_epoch ': self.num_iterations_per_epoch,
        'num_val_iters_per_epoch':self.num_val_iterations_per_epoch,
        'num_epochs': self.num_epochs,
        })

        #  Define a dictionary to store gradients for each layer
        

       
        save_frequency = 1
        self.name = wandb.run.name
        for epoch in range(self.current_epoch, self.num_epochs):
            print(f"Epoch {epoch}: ")
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []

            for batch_id in range(self.num_iterations_per_epoch):
                print(f"Batch_id {batch_id}: ")
                train_outputs.append(self.train_step(next(self.dataloader_train)))

                # Log gradients to WandB (optional)
                for name, grad in self.gradients.items():
                    wandb.log({
                        f"Gradient/{name}_mean": grad.mean().item(),
                        f"Gradient/{name}_max": grad.max().item(),
                        f"Gradient/{name}_min": grad.min().item()
                    })

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        print('next epoch!')
        self.on_train_end()



    