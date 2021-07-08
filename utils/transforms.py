# Import requires libraries.
from monai import transforms
import numpy as np
import torch
from utils.constants import *

# Data augmentation class.
class Transform:
    def __init__(self,augmentation_flipprob,augmentation_affineprob,augmentation_rotate,augmentation_shear,augmentation_scale):
        self.transform_flip = transforms.RandFlipd((const_MODS, const_SEG, const_MASK),prob=augmentation_flipprob, spatial_axis=2)
        self.transform_affine = transforms.RandAffined((const_MODS, const_SEG, const_MASK),
                prob=augmentation_affineprob,
                rotate_range=augmentation_rotate,
                shear_range=augmentation_shear,
                scale_range=augmentation_scale,
                mode=('bilinear','nearest','nearest'),
                padding_mode='zeros')

    def __call__(self, data_dict):
    
        data_dict[const_MASK] = data_dict[const_MASK][np.newaxis,...]
        
        # apply transforms
        data_dict = self.transform_flip(data_dict)
        data_dict = self.transform_affine(data_dict)

        # apply mask and renormalize (inside brain mask)
        data_dict[const_MODS] *= data_dict[const_MASK]
        data_dict[const_MASK] = torch.squeeze(data_dict[const_MASK], dim=0)
        for key in data_dict:
            if type(data_dict[key]) == torch.Tensor:
                data_dict[key] = data_dict[key].numpy()
                
        return data_dict