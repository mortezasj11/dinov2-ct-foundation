import h5py
import numpy as np
from typing import Callable, Optional, Any, Tuple
from io import BytesIO
from PIL import Image

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder






import sys
import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
import os


class CTDataset(ExtendedVisionDataset):
    def __init__(
        self,
        h5_file_path: str,
        patch_size: str = '512,512,3',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        #logger.info("CTDataset is being instantiated.")  # This should appear if CTDataset is used
        super().__init__(root='', transforms=transforms, transform=transform, target_transform=target_transform)

        # Set up directory path and patch size
        self.h5_dir_path = h5_file_path
        self.patch_size = tuple(map(int, patch_size.split(',')))
        
        # Ensure the path is a directory
        if not os.path.isdir(self.h5_dir_path):
            #logger.error(f"h5_file_path '{self.h5_dir_path}' is not a valid directory.")
            raise ValueError(f"h5_file_path '{self.h5_dir_path}' is not a valid directory.")
        
        # Get a list of all .h5 files in the directory
        self.h5_file_paths = [
            os.path.join(self.h5_dir_path, f)
            for f in os.listdir(self.h5_dir_path)
            if f.endswith('.h5')
        ]
        
        # Check that there are .h5 files in the directory
        if not self.h5_file_paths:
            #logger.error(f"No .h5 files found in directory '{self.h5_dir_path}'")
            raise ValueError(f"No .h5 files found in directory '{self.h5_dir_path}'")
        
        #logger.info(f"Found {len(self.h5_file_paths)} .h5 files in directory '{self.h5_dir_path}'")
        
        # Initialize index_map
        # self.index_map = []
        # for h5_file_path in self.h5_file_paths:
        #     with h5py.File(h5_file_path, 'r') as h5_file:
        #         dataset_keys = list(h5_file.keys())
        #         for dataset_key in dataset_keys:
        #             self.index_map.append((h5_file_path, dataset_key, 0))

        self.index_map = []
        for h5_file_path in self.h5_file_paths:
            with h5py.File(h5_file_path, 'r') as h5_file:
                dataset_keys = list(h5_file.keys())
                if dataset_keys:
                    first_key = dataset_keys[0]
                    self.index_map.append((h5_file_path, first_key, 0))
        
        #logger.info(f"Total samples available: {len(self.index_map)}")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def get_image_data(self, index: int) -> Optional[bytes]:
        h5_file_path, dataset_key, _ = self.index_map[index]  # The last value (`sample_index`) is not needed here
        
        with h5py.File(h5_file_path, 'r') as h5_file:
            ct_data = h5_file[dataset_key][...]  # Load the full 3D dataset directly
            #logger.info(f"Loaded full 3D ct_data.shape: {ct_data.shape}")
            
            x, y, z = ct_data.shape
            px, py, pz = self.patch_size

            # Log dimensions for debugging
            #logger.info(f"Sample {index}: dimensions {x, y, z}; Patch size: {px, py, pz}")  #Sample 1841: dimensions (512, 512, 115); Patch size: (512, 512, 3)

            # Check if dimensions are sufficient for the patch size
            if x < px or y < py or z < pz:
                #logger.warning(f"Skipping sample {index} because dimensions {ct_data.shape} are smaller than patch size {self.patch_size}.")
                return None

            # Extract the patch if the sample dimensions are valid
            x_start = np.random.randint(0, x - px + 1)
            y_start = np.random.randint(0, y - py + 1)
            z_start = np.random.randint(0, z - pz + 1)
            image_slice = ct_data[x_start:x_start + px, y_start:y_start + py, z_start:z_start + pz]
            image_slice = np.clip(image_slice, -1000, 200)
            image_slice = (image_slice + 1000) / 1200.0
            logger.info(f"################# image_slice.shape: {image_slice.shape}")

            # Serialize the array to bytes
            buffer = BytesIO()
            np.save(buffer, image_slice)
            buffer.seek(0)
            return buffer.getvalue()


    def get_target(self, index: int) -> Any:
        # Return None as the target
        return None



# In dinov2/data/datasets/__init__.py or similar
# from .ct_dataset import CTDataset
# DATASET_REGISTRY["CTDataset"] = CTDataset


## Testing
# dataset = CTDataset(h5_file_path='path/to/your/file.h5', patch_size=(512, 512, 3))
# image, target = dataset[0]
# print(type(image))  # Should be PIL Image
# print(image.size)   # Should match your patch size
# print(target)       # Should be None