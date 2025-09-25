# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import TargetDecoder, ImageDataDecoder


import logging
# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
import numpy as np
from io import BytesIO
import torch

class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
            #image = torch.from_numpy(image_data).float()
            #image = image.permute(2, 0, 1)
            #logger.info(f"in extend.py 11111 image: {image.size} {image.shape} {image.max()} {self.transforms}")
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()
        #logger.info(f"in extend.py 11111 transform: {self.transforms}")
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        raise NotImplementedError



    # try:
    #     image_data = self.get_image_data(index)
    #     # Retry another index if image_data is None
    #     # if image_data is None:
    #     #     new_index = np.random.randint(0, len(self))
    #     #     return self.__getitem__(new_index)
    #     # Decode image if valid data is returned

    #     #f = BytesIO(image_data)
    #     #image = np.load(f)
    #     image = torch.from_numpy(image_data).float()
    #     image = image.permute(2, 0, 1)
    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     try:
    #         image_data = self.get_image_data(index)
    #         image = ImageDataDecoder(image_data).decode()
    #     except Exception as e:
    #         raise RuntimeError(f"can not read image for sample {index}") from e
    #     target = self.get_target(index)
    #     target = TargetDecoder(target).decode()

    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)

    #     return image, target