import numpy as np
from typing import Callable, Optional, Any, Tuple
from io import BytesIO
import os
import sys
import logging
from torchvision import transforms as ts
import torch
# Setup logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder

#npz_file_paths = list(np.load("/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/All_SlicesNpy.npy"))
#npz_file_paths = list(np.load("/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/All_B1_14_Npz_6chl.npy", allow_pickle=True)) #17M
npz_file_paths = list(np.load("/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/All_Npz_6chl_np1x.npy", allow_pickle=True)) #23M

class MaybeToTensor(ts.ToTensor):
    def __call__(self, pic):
        """
        Args:pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

class CTDataset(ExtendedVisionDataset):
    def __init__(
        self,
        npz_dir_path = None,
        patch_size: str = '512,512,6',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None):
        super().__init__(root='', transforms=transforms, transform=transform, target_transform=target_transform)
        self.patch_size = tuple(map(int, patch_size.split(',')))
        self.index_map = npz_file_paths

    def __len__(self) -> int:
        return len(self.index_map)

    def get_image_data(self, index: int) -> np.ndarray:
        npz_file_path = self.index_map[index]
        try:
            image_mask = np.load(npz_file_path, mmap_mode='r')
            mask_data  = image_mask["mask"]
            image_data = image_mask["array"]
            vmin, vmax, eps = -1000., 150., 0.00005
            image_data = np.clip(image_data, vmin, vmax )
            image_data = np.clip((image_data - vmin) / (vmax-vmin), eps, 1-eps)
            image_data = MaybeToTensor()(image_data)

            mask_data = MaybeToTensor()(mask_data) 
            combined = torch.cat([image_data, mask_data], dim=0) 
            return combined
        except Exception as e:
            logger.warning(f"Error loading image at index {index} from '{npz_file_path}': {e}")
            return self.get_default_image()

    # def get_default_image(self) -> np.ndarray:
    #     return np.ones(self.patch_size, dtype=np.float32)
    def get_default_image(self) -> torch.Tensor:
        default_img = np.ones(self.patch_size, dtype=np.float32)
        default_img = MaybeToTensor()(default_img)
        return default_img

    def get_target(self, index: int) -> Any:
        # Return None as the target
        return None


