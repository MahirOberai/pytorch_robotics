import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, img_dir, has_gt):
        """
        In:
            img_dir: string, path of train, val or test folder.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        Hint:
            Check __getitem__() and add more instance variables to initialize what you need in this method.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.img_dir = img_dir
        self.has_gt = has_gt
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_rgb, std=std_rgb)])
        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset.
        self.dataset_length = sum(1 for f in os.listdir(self.img_dir+'rgb') if '.png' in f)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_path = os.path.join(self.img_dir+'rgb/', str(idx)+'_rgb.png')
        gt_path = os.path.join(self.img_dir+'gt/', str(idx)+'_gt.png')

        rgb_img = image.read_rgb(rgb_path)
        gt_mask = image.read_mask(gt_path)

        transformed_rgb = self.transform(rgb_img)
        if self.has_gt is False:
            sample = {'input': transformed_rgb}
        else:
            transformed_gt = torch.LongTensor(gt_mask)
            sample = {'input': transformed_rgb, 'target': transformed_gt}

        return sample
