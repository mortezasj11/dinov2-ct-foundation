# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

import torch
logger = logging.getLogger("dinov2")

import random

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # gg      geometric_augmentation_global
        self.gg = transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        # gl       geometric_augmentation_local
        self.gl = transforms.Compose(
            [
                transforms.RandomRotation(degrees=60),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)],   #[transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01)],    hue is the problem!!
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0, radius_min=0.1, radius_max=1.0)

        # global_transfo2_extra = transforms.Compose(
        #     [
        #         GaussianBlur(p=0.1, radius_min=0.1, radius_max=1.0),
        #         transforms.RandomSolarize(threshold=0.1, p=0.2),
        #     ]
        # )
        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1, radius_min=0.1, radius_max=1.0),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5, radius_min=0.1, radius_max=1.0)

        # normalization
        self.normalize = transforms.Compose(
            [
                #transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.ag1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.ag2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.al = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def gl_image_and_mask(self, image, mask):
            combined = torch.cat([image, mask], dim=0)  # Combined shape: [C+C, H, W]
            combined_aug = self.gl(combined) # Apply gl
            image_aug = combined_aug[:image.shape[0], :, :]
            mask_aug = combined_aug[image.shape[0]:, :, :]  
            #if random.random() < 0.55 and torch.sum(mask_aug)>3*2000: 
            if random.random() < 0.333 : 
                image_aug = image_aug * mask_aug
            return image_aug

    def __call__(self, image_mask):
        image = image_mask[:3, :, :]
        mask = image_mask[3:, :, :]
        output = {}

        # global crops:
        im1_base = self.gg(image)
        global_crop_1 = self.ag1(im1_base)

        im2_base = self.gg(image)
        global_crop_2 = self.ag2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        ## 1. local crops (original)
        # local_crops = [self.al(self.gl(image)) for _ in range(self.local_crops_number)]

        ## 2. local crops (safe with while)
        # local_crops, crop_n = [], 1
        # while len(local_crops) < self.local_crops_number:
        #     aug_image = self.gl_image_and_mask(image, mask)
        #     aug_image = self.al(aug_image)
        #     if torch.isfinite(aug_image).all():
        #         local_crops.append(aug_image)
        #         crop_n += 1

        ## 3. local crops (safe as hue=0 with for)
        # local_crops = []
        # for i in range(self.local_crops_number):
        #     aug_image = self.gl_image_and_mask(image, mask)
        #     aug_image = self.al(aug_image)
        #     local_crops.append(aug_image)

        ## 4. local crops (safe as hue=0 with for)
        local_crops = [self.al(self.gl_image_and_mask(image, mask)) for _ in range(self.local_crops_number)]

        output["local_crops"] = local_crops
        output["offsets"] = ()

        # import random
        # rand_num = random.randint(1000, 9999)
        # torch.save(output, f"/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/zzz_aug_{rand_num}.pt")

        return output
