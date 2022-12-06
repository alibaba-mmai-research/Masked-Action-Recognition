#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# From https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py

# MIT License

# Copyright (c) 2019 Tengda Han

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Transformations. """
import torch
import math

import torchvision.transforms._functional_video as F
from torchvision.transforms import Lambda, Compose
import random
import numbers

# TODO: clean up the codes

class ColorJitter(object):
    """
    Modified from https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py.
    Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale (float): possibility to transform the video to grayscale. 
            Should have a value range of [0, 1]
        consistent  (bool): indicates whether or not to keep all the color transformations consistent for all the frames.
        shuffle     (bool): indicates whether or not to shuffle the sequence of the augmentations.
        gray_first  (bool): indicates whether or not to put grayscale transform first.
    """
    def __init__(
        self, brightness=0, contrast=0, saturation=0, hue=0, grayscale=0, consistent=False, shuffle=True, gray_first=True
    ):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        
        self.grayscale = grayscale
        self.consistent = consistent
        self.shuffle = shuffle
        self.gray_first = gray_first

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_transform(self, T, device):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Arg:
            T (int): number of frames. Used when consistent = False.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if self.brightness is not None:
            if self.consistent:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            else:
                brightness_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda frame: adjust_brightness(frame, brightness_factor)))
        
        if self.contrast is not None:
            if self.consistent:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            else:
                contrast_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda frame: adjust_contrast(frame, contrast_factor)))
        
        if self.saturation is not None:
            if self.consistent:
                saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            else:
                saturation_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda frame: adjust_saturation(frame, saturation_factor)))
        
        if self.hue is not None:
            if self.consistent:
                hue_factor = random.uniform(self.hue[0], self.hue[1])
            else:
                hue_factor = torch.empty([T, 1, 1], device=device).uniform_(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda frame: adjust_hue(frame, hue_factor)))

        if self.shuffle:
            random.shuffle(transforms)
        
        if random.uniform(0, 1) < self.grayscale:
            gray_transform = Lambda(lambda frame: rgb_to_grayscale(frame))
            if self.gray_first: 
                transforms.insert(0, gray_transform)
            else:
                transforms.append(gray_transform)
        
        transform = Compose(transforms)

        return transform

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        
        raw_shape = clip.shape #(C, T, H, W)
        device = clip.device
        T = raw_shape[1]
        transform = self._get_transform(T, device)
        clip = transform(clip)
        assert clip.shape == raw_shape
        return clip #(C, T, H, W)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string

def _is_tensor_a_torch_image(input):
    return input.ndim >= 2

def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)

def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
        Args:
            clip (torch.tensor): Size is (T, H, W, C)
        Return:
            clip (torch.tensor): Size is (T, H, W, C)
    """
    orig_dtype = img.dtype
    rgb_convert = torch.tensor([0.299, 0.587, 0.114])
    
    assert img.shape[0] == 3, "First dimension need to be 3 Channels"
    if img.is_cuda:
        rgb_convert = rgb_convert.to(img.device)
    
    img = img.float().permute(1,2,3,0).matmul(rgb_convert).to(orig_dtype)
    return torch.stack([img, img, img], 0)

def _rgb2hsv(img):
    r, g, b = img.unbind(0)

    maxc, _ = torch.max(img, dim=0)
    minc, _ = torch.min(img, dim=0)
    
    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))

def _hsv2rgb(img):
    l = len(img.shape)
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    if l == 3:
        tmp = torch.arange(6)[:, None, None]
    elif l == 4:
        tmp = torch.arange(6)[:, None, None, None]
    
    if img.is_cuda:
        tmp = tmp.to(img.device)

    mask = i == tmp #(H, W) == (6, H, W)

    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3)) #(3, 6, H, W)

    if l == 3:
        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4) #(C, H, W)
    elif l == 4:
        return torch.einsum("itjk, xitjk -> xtjk", mask.to(dtype=img.dtype), a4) #(C, T, H, W)

def adjust_brightness(img, brightness_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)

def adjust_contrast(img, contrast_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    
    mean = torch.mean(rgb_to_grayscale(img).to(torch.float), dim=(-4, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)

def adjust_saturation(img, saturation_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    if isinstance(hue_factor, float) and  not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    elif isinstance(hue_factor, torch.Tensor) and not ((-0.5 <= hue_factor).sum() == hue_factor.shape[0] and (hue_factor <= 0.5).sum() == hue_factor.shape[0]):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(0)
    h += hue_factor
    h = h % 1.0
    img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj

class CustomResizedCropVideo(object):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            interpolation_mode="bilinear",
            mode=1
    ):
        # mode how many clips return
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.mode = mode

    def get_one_crop(self, clip):
        scale = random.uniform(*self.scale)

        # Get the crop size for the scale cropping
        _, _, image_height, image_width = clip.shape

        min_length = min(image_width, image_height)
        crop_size = int(min_length * scale)

        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2
        th = crop_size
        tw = crop_size

        positions = []

        x1 = center_x - box_half
        y1 = center_y - box_half
        x2 = center_x + box_half
        y2 = center_y + box_half
        positions.append([x1, x2, y1, y2])

        crop = F.resized_crop(clip, y1, x1, th, tw, self.size, self.interpolation_mode)
        crops = torch.unsqueeze(crop, 0)
        return crops

    def get_three_crop(self, clip, if_flip=False):
        # Choose a random position and random scale
        scale = random.uniform(*self.scale)
        crop_position = ['c', 'l', 'r']

        # Get the crop size for the scale cropping
        _, _, image_height, image_width = clip.shape

        min_length = min(image_width, image_height)
        crop_size = int(min_length * scale)

        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2
        th = crop_size
        tw = crop_size

        positions = []
        # Do the scale cropping at the chosen position
        if 'c' in crop_position:
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
            positions.append([x1, x2, y1, y2])
        if 'l' in crop_position:
            x1 = 0
            y1 = center_y - box_half
            x2 = crop_size
            y2 = center_y + box_half
            positions.append([x1, x2, y1, y2])
        if 'r' in crop_position:
            x1 = image_width - crop_size
            y1 = center_y - box_half
            x2 = image_width
            y2 = center_y + box_half
            positions.append([x1, x2, y1, y2])
        else:
            raise ValueError("Crop position must be 1 of c, l, r")

        # print(positions)
        crops = []
        for [x1, x2, y1, y2] in positions:
            crop = F.resized_crop(clip, y1, x1, th, tw, self.size, self.interpolation_mode)
            crops.append(crop)
            if if_flip:
                crops.append(F.hflip(crop))
            
        return torch.stack(crops, dim=0)

    def get_five_crop(self, clip, if_flip=False):
        # Choose a random position and random scale
        scale = random.uniform(*self.scale)
        crop_position = ['c', 'tl', 'tr', 'bl', 'br']

        # Get the crop size for the scale cropping
        _, _, image_height, image_width = clip.shape

        min_length = min(image_width, image_height)
        crop_size = int(min_length * scale)
        th = crop_size
        tw = crop_size

        positions = []
        # Do the scale cropping at the chosen position
        if 'c' in crop_position:
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
            positions.append([x1, x2, y1, y2])
        if 'tl' in crop_position:
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
            positions.append([x1, x2, y1, y2])
        if 'tr' in crop_position:
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
            positions.append([x1, x2, y1, y2])
        if 'bl' in crop_position:
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
            positions.append([x1, x2, y1, y2])
        if 'br' in crop_position:
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height
            positions.append([x1, x2, y1, y2])

        crops = []
        for [x1, x2, y1, y2] in positions:
            crop = F.resized_crop(clip, y1, x1, th, tw, self.size, self.interpolation_mode)
            crops.append(crop)
            if if_flip:
                crops.append(F.hflip(crop))

        return torch.stack(crops, dim=0)       

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        if self.mode == 1:
            return self.get_one_crop(clip)

        if self.mode == 3:
            return self.get_three_crop(clip)

        if self.mode == 5:
            return self.get_five_crop(clip)

        if self.mode == 6:
            return self.get_three_crop(clip, if_flip=True)

        if self.mode == 10:
            return self.get_five_crop(clip, if_flip=True)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(size={0}, interpolation_mode={1}, scale={2})'.format(
                   self.size, self.interpolation_mode, self.scale
               )

class AutoResizedCropVideo(object):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            interpolation_mode="bilinear",
            mode = "cc"
    ):
        # mode how many clips return
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.mode = mode
        self.idx = 0

    def set_spatial_index(self, idx):
        self.idx = idx

    def get_crop(self, clip):
        crop_mode = self.mode[self.idx*2:self.idx*2+2]

        scale = random.uniform(*self.scale)

        # Get the crop size for the scale cropping
        _, _, image_height, image_width = clip.shape

        min_length = min(image_width, image_height)
        crop_size = int(min_length * scale)

        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2
        th = crop_size
        tw = crop_size

        if crop_mode == "cc":
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif crop_mode == "ll":
            x1 = 0
            y1 = center_y - box_half
            x2 = crop_size
            y2 = center_y + box_half
        elif crop_mode == "rr":
            x1 = image_width - crop_size
            y1 = center_y - box_half
            x2 = image_width
            y2 = center_y + box_half
        elif crop_mode == "tl":
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif crop_mode == "tr":
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif crop_mode == "bl":
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif crop_mode == "br":
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        crop = F.resized_crop(clip, y1, x1, th, tw, self.size, self.interpolation_mode)
        return crop

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        if self.idx == -1:
            # return self.get_random_crop(clip)
            pass
        else:
            return self.get_crop(clip)

class KineticsResizedCrop(object):
    def __init__(
        self,
        short_side_range,
        crop_size,
        num_spatial_crops=1,
    ):  
        self.idx = -1
        self.short_side_range = short_side_range
        self.crop_size = int(crop_size)
        self.num_spatial_crops = num_spatial_crops
    
    def _get_controlled_crop(self, clip):
        _, _, clip_height, clip_width = clip.shape

        length = self.short_side_range[0]

        if clip_height < clip_width:
            new_clip_height = int(length)
            new_clip_width = int(clip_width / clip_height * new_clip_height)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        else:
            new_clip_width = int(length)
            new_clip_height = int(clip_height / clip_width * new_clip_width)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        if self.num_spatial_crops == 1:
            x = x_max // 2
            y = y_max // 2
        elif self.num_spatial_crops == 3:
            if self.idx == 0:
                if new_clip_width == length:
                    x = x_max // 2
                    y = 0
                elif new_clip_height == length:
                    x = 0
                    y = y_max // 2
            elif self.idx == 1:
                x = x_max // 2
                y = y_max // 2
            elif self.idx == 2:
                if new_clip_width == length:
                    x = x_max // 2
                    y = y_max
                elif new_clip_height == length:
                    x = x_max
                    y = y_max // 2
        return new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]

    def _get_random_crop(self, clip):
        _, _, clip_height, clip_width = clip.shape

        if clip_height < clip_width:
            new_clip_height = int(random.uniform(*self.short_side_range))
            new_clip_width = int(clip_width / clip_height * new_clip_height)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        else:
            new_clip_width = int(random.uniform(*self.short_side_range))
            new_clip_height = int(clip_height / clip_width * new_clip_width)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        x = int(random.uniform(0, x_max))
        y = int(random.uniform(0, y_max))
        return new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]

    def set_spatial_index(self, idx):
        self.idx = idx

    def __call__(self, clip):
        if self.idx == -1:
            return self._get_random_crop(clip)
        else:
            return self._get_controlled_crop(clip)

class RandomResizedCropVideo(object):
    def __init__(
        self,
        resize,
        crop_size
    ):
        if isinstance(resize, tuple):
            assert len(resize) == 2, "size should be tuple (height, width)"
            self.resize = resize
        else:
            self.resize = (resize, resize)

        self.crop_size = crop_size

    def get_crop_params(self, clip):
        c,t,h,w = clip.shape
        if h == self.crop_size and w == self.crop_size:
            return 0, 0
        
        i = random.randint(0, h-self.crop_size)
        j = random.randint(0, w-self.crop_size)
        return i, j


    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        clip = torch.nn.functional.interpolate(
            clip, size=self.resize, mode="bilinear"
        )
        i,j = self.get_crop_params(clip)
        return clip[..., i:i+self.crop_size, j:j+self.crop_size]

    def __repr__(self):
        return self.__class__.__name__ + \
            '(resize={0}, crop_size={1})'.format(
                self.resize, self.crop_size
            )


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = "bilinear"

    def __call__(self, clip):
        c,t,h,w = clip.shape
        im_size = (w, h)
        w, h, j, i = self._sample_crop_size(im_size)
        assert w + j <= im_size[0] and h + i <= im_size[1]
        clip = F.resized_crop(clip, i, j, h, w, self.input_size, self.interpolation)
        return clip

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class TemporalInconsistentRandomResizedCropVideo(object):
    def __init__(
        self,
        crop_size,
        temporal_patch_size,
        inconsistent_type,
        jitter_scale,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(crop_size, tuple):
            assert len(crop_size) == 2, "size should be tuple (height, width)"
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)
        self.temporal_patch_size = temporal_patch_size
        self.jitter_scale = jitter_scale
        self.inconsistent_type = inconsistent_type
        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def get_params_jitter(self, clip, scale, ratio):
        t, width, height = clip.shape[-3], clip.shape[-1], clip.shape[-2]
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i_list, j_list, h_list, w_list = [], [], [], []
                for _ in range(t//self.temporal_patch_size):
                    h_list.append(h)
                    w_list.append(w)
                    if len(i_list) == 0:
                        i_list.append(torch.randint(0, height - h + 1, size=(1,)).item())
                        j_list.append(torch.randint(0, width - w + 1, size=(1,)).item())
                    else:
                        i_list.append(torch.randint(max(0, i_list[-1] - self.jitter_scale), min(height - h + 1, i_list[-1] + self.jitter_scale), size=(1,)).item())
                        j_list.append(torch.randint(max(0, j_list[-1] - self.jitter_scale), min(width - w + 1, j_list[-1] + self.jitter_scale), size=(1,)).item())
                return i_list, j_list, h_list, w_list

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = [(height - h) // 2 for _ in range(t//self.temporal_patch_size)]
        j = [(width - w) // 2 for _ in range(t//self.temporal_patch_size)]
        h = [h for _ in range(t//self.temporal_patch_size)]
        w = [w for _ in range(t//self.temporal_patch_size)]
        return i, j, h, w

    def get_params_simple(self, clip, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        t, width, height = clip.shape[-3], clip.shape[-1], clip.shape[-2]
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_random(self, clip, scale, ratio):
        t, width, height = clip.shape[-3], clip.shape[-1], clip.shape[-2]
        i_list, j_list, h_list, w_list = [], [], [], []
        for _ in range(t // self.temporal_patch_size):
            i, j, h, w = self.get_params_simple(clip, scale, ratio)
            i_list.append(i)
            j_list.append(j)
            h_list.append(h)
            w_list.append(w)
        return i_list, j_list, h_list, w_list

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        if self.inconsistent_type == 'jitter':
            i_list, j_list, h_list, w_list = self.get_params_jitter(clip, self.scale, self.ratio)
        elif self.inconsistent_type == 'random':
            i_list, j_list, h_list, w_list = self.get_params_random(clip, self.scale, self.ratio)
        cropped_clips = []
        for idx, (i, j, h, w) in enumerate(zip(i_list, j_list, h_list, w_list)):
            cropped_clips.append(F.resized_crop(clip[:, idx*self.temporal_patch_size:(idx+1)*self.temporal_patch_size], 
                                                i, j, h, w, 
                                                self.crop_size, 
                                                self.interpolation_mode))
        return torch.cat(cropped_clips, dim=1)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2}, ratio={3})'.format(
                self.crop_size, self.interpolation_mode, self.scale, self.ratio
            )


class BatchAugmentation(object):
    def __init__(self, transform_list):
        self.transform = Compose(transform_list)
    
    def __call__(self, video):
        assert len(video.shape) == 5
        transformed_list = []
        for i in range(video.shape[0]):
            transformed_list.append(self.transform(video[i]))
        return torch.stack(transformed_list, dim=0)


def imshow_raw(video, name='raw_video_frame'):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    print(video.dim(), video.shape, video.size())
    for i in range(3):
        for j in range(3):
            print(video[i*3+j, 0].shape)
            ax = fig.add_subplot(3, 3, i*3+j+1)
            plt.imshow(video[i*3+j, 0].numpy())
            # plt.imsave('/home/admin/workspace/raw_video_frame_{}.jpg'.format(i), video[i].numpy())

    plt.savefig('/home/admin/workspace/{}.jpg'.format(name))

def imshow(video):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=video.dtype, device=video.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=video.dtype, device=video.device)
    video.mul_(std[None :, None, None, None]).add_(mean[None :, None, None, None])
    video = (video*255.0).permute(0, 2, 3, 4, 1).int()
    imshow_raw(video, 'transform')

def test():
    from torchvision.transforms import Compose
    import torchvision.transforms._transforms_video as transforms
    import decord
    from decord import VideoReader
    from decord import cpu, gpu
    decord.bridge.set_bridge('torch')
    vr = VideoReader('/home/admin/workspace/shared/public/gongbao.yq/ucf101/videos/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi', ctx=cpu(0))
    video = vr.get_batch([101, 5, 109, 13, 117, 21, 125, 29, 133])
    imshow_raw(video)
    print("video.shape:{}".format(video.shape)) #T, H, W, C

    transform_1 = Compose([
        ColorJitter(
            brightness=0.5, # 0.5
            contrast=0.5, # 0.5 
            saturation=0.5, # 0.5
            hue=0,
            grayscale=0.25, # 0.25
            consistent=True,
            shuffle=False,
            gray_first=False,
            is_split=False #only for gray
        ),
        transforms.ToTensorVideo(),
        transforms.RandomResizedCropVideo(
            size=112,
            scale=(0.375, 0.75),
            ratio=(0.857142857142857, 1.1666666666666667)
        ),
        transforms.RandomHorizontalFlipVideo(),
        transforms.NormalizeVideo(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            inplace=True
        )
    ])

    transform_2 = Compose([
        transforms.ToTensorVideo(),
        CustomResizedCropVideo(
            size=112,
            scale=(0.875, 0.875),
            mode=10,
        ),
    ])

    video = transform_2(video)
    imshow(video)
    print("after transform, video.shape:{}".format(video.shape)) #C, T, H, W

if __name__ == '__main__':
    test()