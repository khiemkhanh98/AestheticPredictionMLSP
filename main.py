import numpy as np
import cv2
import torch
import os


def mirror_reflection(img):
    # function for flip augmentation
    img = np.flip(img, 1)
    return img


def multiple_crops(img, crop_shape, do_flips=False):
    # In the article there were 8 augmentations: crops along corners + flips combinations

    assert (crop_shape[0] < img.shape[1]) and (crop_shape[1] < img.shape[2]), \
        'Crop shape must be less than rescale size!'

    img_dl = img[:, 0:crop_shape[0], 0:crop_shape[1]]  # down left corner
    img_dr = img[:, 0:crop_shape[0], -crop_shape[1]:]  # down right corner
    img_ul = img[:, -crop_shape[0]:, 0:crop_shape[1]]  # up left corner
    img_ur = img[:, -crop_shape[0]:, -crop_shape[1]:]  # up right corner

    if do_flips:
        res = np.stack([img_dl,
                        img_dr,
                        img_ul,
                        img_ur,
                        mirror_reflection(img_dl),
                        mirror_reflection(img_dr),
                        mirror_reflection(img_ul),
                        mirror_reflection(img_ur)])
    else:
        res = np.stack([img_dl,
                        img_dr,
                        img_ul,
                        img_ur])
    # Result is batch of 4-8 images

    return res


class DataLoader(torch.nn.Module):
    def __init__(self, path_to_data: str,
                 rescale_size, crop_size):
        '''
        :param path_to_data: path to directory with data
        :param rescale_size: size to which image must be rescaled to
        :param crop_size: size to which rescaled image must be cropped to
        '''

        super(DataLoader, self).__init__()
        self.path_to_data = path_to_data
        self.rescale_size = rescale_size
        self.crop_size = crop_size

        assert os.path.isdir(path_to_data), \
            'Path to data must be a directory with images!'
        assert (self.crop_size[0] < rescale_size[0]) and (self.crop_size[1] < rescale_size[1]), \
            'Crop shape must be less than rescale size!'

        self.to_preprocess = os.listdir(path_to_data)

    def __len__(self):
        return len(self.to_preprocess)

    def __getitem__(self, i):
        file = self.to_preprocess[i]
        # If there are some directories inside directory with images, then ignore it
        path = os.path.join(self.path_to_data, file)
        if os.path.isdir(path):
            print("Path {} is assigned to dir".format(path))
            return None
        # If image file is broken or there are files that are not images, then ignore it
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except:
            return None

        img = cv2.resize(img, self.rescale_size)
        img = np.array(img)
        if len(img.shape) == 3:
            img = np.array(img).transpose(2, 0, 1)
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            print("Image {} is broken or is not image".format(path))
            return None

        pack = multiple_crops(img, self.crop_size)

        return pack
