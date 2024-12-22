import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def aug_transforms():
    return [
        # A.VerticalFlip(p=1),
        # A.HorizontalFlip(p=1),
        A.Rotate(limit=(-90, 90), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None,
                 always_apply=False, p=1),
        # A.ElasticTransform(alpha=20, sigma=50, alpha_affine=8,
        #                    interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=None,
        #                    mask_value=None, always_apply=False, approximate=False, p=1),
        # A.Crop(0, 40, 100, 144),
        # A.Crop(40, 40, 140, 144),
        # A.Crop(0, 0, 144, 144),
        # A.Crop(10, 20, 144, 144)

        # A.RandomBrightnessContrast()
        # A.GridDistortion(num_steps=20, distort_limit=0.2, interpolation=cv2.INTER_NEAREST,
        #                     border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None,
        #                     always_apply=False, p=1)

    ]


aug = A.Compose(aug_transforms())
np.random.seed(7)

# Read an image with OpenCV and convert it to the RGB colorspace
im_pth = r'multiClass_DS_inst\train'
files_list = os.listdir(im_pth)
msk_list = [fname for fname in files_list if 'msk' in fname and 'aug' not in fname]
im_list = [fname for fname in files_list if 'msk' not in fname and 'aug' not in fname]

for i in range(0, len(msk_list)):
    image = np.load(os.path.join(im_pth, im_list[i]))
    mask = np.load(os.path.join(im_pth, msk_list[i]))
    # print(os.path.join(im_pth, im_list[i]))
    # Augment an image
    for j in range(1):
        transformed = aug(image=image, mask=mask)
        transformed_image, transformed_mask = transformed["image"], transformed["mask"]
        aug_msk_name = 'rot_aug_' + msk_list[i].split('msk')[0] + str(j) + '_msk.npy'
        aug_im_name = 'rot_aug_' + im_list[i].split('.npy')[0] + '_' + str(j) + '.npy'
        np.save(os.path.join(im_pth, aug_msk_name), transformed_mask)
        np.save(os.path.join(im_pth, aug_im_name), transformed_image)
        print(aug_msk_name, '\t', aug_im_name)