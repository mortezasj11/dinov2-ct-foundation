import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import logging

logger = logging.getLogger(__name__)

class DataAugmentationDINO(object):
    # Augmentation options (class attributes)
    use_random_resized_crop = True
    use_random_horizontal_flip = True
    use_color_jitter = True
    use_random_grayscale = True
    use_gaussian_blur = True
    use_random_solarize = True

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
        logger.info(f"use_random_resized_crop: {self.use_random_resized_crop}")
        logger.info(f"use_random_horizontal_flip: {self.use_random_horizontal_flip}")
        logger.info(f"use_color_jitter: {self.use_color_jitter}")
        logger.info(f"use_random_grayscale: {self.use_random_grayscale}")
        logger.info(f"use_gaussian_blur: {self.use_gaussian_blur}")
        logger.info(f"use_random_solarize: {self.use_random_solarize}")
        logger.info("###################################")

    def random_resized_crop(self, image, size, scale, ratio):
        if self.use_random_resized_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=scale, ratio=ratio
            )
            return F.resized_crop(image, i, j, h, w, (size, size), InterpolationMode.BICUBIC)
        else:
            # If not using random resized crop, resize the image directly
            return F.resize(image, (size, size), interpolation=InterpolationMode.BICUBIC)

    def random_horizontal_flip(self, image, p=0.5):
        if self.use_random_horizontal_flip and torch.rand(1) < p:
            return F.hflip(image)
        return image

    def color_jittering(self, image):
        if self.use_color_jitter and torch.rand(1) < 0.8:
            # Define the jitter parameters
            brightness = 0.4
            contrast = 0.4
            saturation = 0.2
            hue = 0.1

            # Randomly apply brightness, contrast, saturation, hue adjustments
            transforms_list = []

            if brightness > 0:
                brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
                transforms_list.append(lambda img: F.adjust_brightness(img, brightness_factor))

            if contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
                transforms_list.append(lambda img: F.adjust_contrast(img, contrast_factor))

            if saturation > 0:
                saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
                transforms_list.append(lambda img: F.adjust_saturation(img, saturation_factor))

            if hue > 0:
                hue_factor = random.uniform(-hue, hue)
                transforms_list.append(lambda img: F.adjust_hue(img, hue_factor))

            random.shuffle(transforms_list)
            for func in transforms_list:
                image = func(image)

        if self.use_random_grayscale and torch.rand(1) < 0.2:
            image = F.rgb_to_grayscale(image, num_output_channels=3)
        return image

    def gaussian_blur(self, image, p=0.5):
        if self.use_gaussian_blur and torch.rand(1) < p:
            sigma = random.uniform(0.1, 2.0)
            kernel_size = int(0.1 * min(image.shape[-2:])) // 2 * 2 + 1
            kernel_size = max(3, kernel_size)  # Ensure kernel_size >= 3
            return F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        return image

    def random_solarize(self, image, threshold=0.5, p=0.2):
        if self.use_random_solarize and torch.rand(1) < p:
            return torch.where(image >= threshold, 1.0 - image, image)
        return image

    def __call__(self, image):
        output = {}

        # Global crop 1
        im1 = self.random_resized_crop(
            image, self.global_crops_size, self.global_crops_scale, (3. / 4., 4. / 3.)
        )
        im1 = self.random_horizontal_flip(im1)
        im1 = self.color_jittering(im1)
        im1 = self.gaussian_blur(im1, p=1.0)
        im1 = F.normalize(im1, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])

        # Global crop 2
        im2 = self.random_resized_crop(
            image, self.global_crops_size, self.global_crops_scale, (3. / 4., 4. / 3.)
        )
        im2 = self.random_horizontal_flip(im2)
        im2 = self.color_jittering(im2)
        im2 = self.gaussian_blur(im2, p=0.1)
        im2 = self.random_solarize(im2, threshold=0.5, p=0.2)
        im2 = F.normalize(im2, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])

        output["global_crops"] = [im1, im2]
        output["global_crops_teacher"] = [im1, im2]

        # Local crops
        local_crops = []
        for _ in range(self.local_crops_number):
            local_im = self.random_resized_crop(
                image, self.local_crops_size, self.local_crops_scale, (3. / 4., 4. / 3.)
            )
            local_im = self.random_horizontal_flip(local_im)
            local_im = self.color_jittering(local_im)
            local_im = self.gaussian_blur(local_im, p=0.5)
            local_im = F.normalize(local_im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            local_crops.append(local_im)
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
