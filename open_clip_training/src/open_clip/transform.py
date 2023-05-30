import random
from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_erosion

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img

class Erosion(nn.Module):

    def __init__(self, ksize_list=(3, 7, 11, 15, 17, 21), mean=(124, 116, 103)):
        super(Erosion, self).__init__()
        self.ksize_list = ksize_list
        self.mean = np.array(mean).astype(np.uint8)

    def forward(self, img):
        ksize = random.choice(self.ksize_list)
        imarray = np.array(img)
        mask = (imarray == self.mean).all(axis=2)
        mask = ~mask
        mask = mask.astype(int)
        erosion_mask = binary_erosion(mask, structure=np.ones((ksize,ksize))).astype(mask.dtype)
        imarray[erosion_mask == 0] = self.mean
        img = Image.fromarray(imarray)
        return img



def _convert_to_rgb(image):
    return image.convert('RGB')

def _convert_to_rgb_w_mask(inp):
    image, mask = inp
    return image.convert('RGB'), mask

def _to_tensor_w_mask(inp):
    image, mask = inp
    return F.to_tensor(image), mask

class Maskget(nn.Module):

    def __init__(self, mean=(124, 116, 103)):
        super(Maskget, self).__init__()
        self.mean = np.array(mean).astype(np.uint8)

    def forward(self, img):
        imarray = np.array(img)
        mask = (imarray == self.mean).all(axis=2)
        mask = ~mask
        mask = mask.astype(int)
        img = Image.fromarray(imarray)
        return img, mask

class Normalize_w_mask(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, inp):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor, mask = inp
        return F.normalize(tensor, self.mean, self.std, self.inplace), mask

def _normalize_w_mask(image, mask):
    return F.to_tensor(image), mask


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        scale: Optional[Tuple[float, ...]] = None,
        erosion: bool = False,
        with_mask: bool = False,
):
    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    scale = scale or (0.9, 1.0)
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        default_transform = Compose([
            RandomResizedCrop(image_size, scale=scale, interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,])
        if with_mask:
            default_transform = Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                Maskget(),
                _convert_to_rgb_w_mask,
                _to_tensor_w_mask,
                Normalize_w_mask(mean=mean, std=std),
            ])
        return default_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        if not with_mask:
            transforms.extend([
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
        else:
            transforms.extend([
                Maskget(),
                _convert_to_rgb_w_mask,
                _to_tensor_w_mask,
                Normalize_w_mask(mean=mean, std=std),
            ])
        return Compose(transforms)
