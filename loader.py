from PIL import ImageFilter
import random

# https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        return [self.base_transform(img), self.base_transform(img)]

class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        s = random.uniform(*self.sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=s))
