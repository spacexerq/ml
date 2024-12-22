import numpy as np
import skimage.color

def class_binarization(multiclass_mask, class_label):
    binary_mask = np.zeros((multiclass_mask.shape[0], multiclass_mask.shape[1]), dtype=np.int32)

    for i in range(multiclass_mask.shape[0]):
        for j in range(multiclass_mask.shape[1]):
            if np.any(multiclass_mask[i, j] == class_label):
                binary_mask[i, j] = 1
            else:
                binary_mask[i, j] = 0
    return binary_mask

def mask_on_image(im, binary_mask, color=(255, 0, 0)):
    colored_mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=im.dtype)
    colored_mask[:, :, 0], colored_mask[:, :, 1], colored_mask[:, :, 2] = binary_mask, binary_mask, binary_mask
    colored_im = skimage.color.gray2rgb(im/np.max(im))
    colored_mask = colored_mask * color
    colored_im = colored_im + colored_mask

    return colored_im


