import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import logging
logger = logging.getLogger("dinov2")



# import logging
# # Configure the logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

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

    def random_resized_crop(self, image, size, scale, ratio):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=scale, ratio=ratio
        )
        return F.resized_crop(image, i, j, h, w, (size, size), InterpolationMode.BICUBIC)

    def random_horizontal_flip(self, image, p=0.5):
        if torch.rand(1) < p:
            return F.hflip(image)
        return image

    def gaussian_blur(self, image, p=0.5):
        if torch.rand(1) < p:
            sigma = float(torch.rand(1).item() * 1.9 + 0.1)
            kernel_size = int(0.1 * min(image.shape[1:])) // 2 * 2 + 1
            return F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        return image

    def __call__(self, image):
        output = {}

        # Global crops
        im1 = self.random_resized_crop(
            image, self.global_crops_size, self.global_crops_scale, (3./4., 4./3.)
        )
        im1 = self.random_horizontal_flip(im1)
        im1 = self.gaussian_blur(im1, p=1.0)
        im1 = F.normalize(im1, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])

        im2 = self.random_resized_crop(
            image, self.global_crops_size, self.global_crops_scale, (3./4., 4./3.)
        )
        im2 = self.random_horizontal_flip(im2)
        im2 = self.gaussian_blur(im2, p=0.1)
        # Apply additional transformations if needed...
        im2 = F.normalize(im2, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])

        output["global_crops"] = [im1, im2]
        output["global_crops_teacher"] = [im1, im2]

        # Local crops
        local_crops = []
        for _ in range(self.local_crops_number):
            local_im = self.random_resized_crop(
                image, self.local_crops_size, self.local_crops_scale, (3./4., 4./3.)
            )
            local_im = self.random_horizontal_flip(local_im)
            local_im = self.gaussian_blur(local_im, p=0.5)
            local_im = F.normalize(local_im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            local_crops.append(local_im)
        output["local_crops"] = local_crops
        output["offsets"] = ()
        #logger.info(f"in augmentation output[global_crops]: {output['global_crops'][0].shape} {output['global_crops']} ")  torch.Size([3, 512, 512])
        #logger.info(f"in augmentation output[global_crops_teacher]: {output['global_crops_teacher'][0].shape} {output['global_crops_teacher']} ")

        return output