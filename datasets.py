from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
import pandas as pd


class ImageDataset(Dataset):

    '''Dataset object for images'''

    def __init__(self, image_path, image_size, transforms=None):

        self.image_size = image_size
        self.image_path = image_path
        self.transforms = transforms

        # get all image files
        self.images = list(Path(image_path).rglob('*.jpg'))

    def __getitem__(self, idx):

        # read as PIL image
        img = Image.open(self.images[idx])

        # if no transforms, PIL to tensor
        if not self.transforms:
            img = transforms.ToTensor()(img)
        else:
            img = self.transforms(img)
        
        return img

    def __len__(self):

        return len(self.images)


###############################################################################


def load_lsun(images_dir='data/lsun', image_size=128):

    '''Load the LSUN dataset. The original dataset is too large (43GB). 
    Use a 20% sample downloaded from kaggle.com/jhoward/lsun_bedroom/data.'''

    # resize images
    # convert from PIL image to tensor
    # normalize all channels with mean 0.5 and std 0.5
    img_transforms = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                                 std = (0.5, 0.5, 0.5))
            ])

    lsun_dataset = ImageDataset(images_dir, image_size, img_transforms)

    # return dataset
    # pass to DataLoader
    return lsun_dataset


def load_celeba(images_dir='data/img_align_celeba', image_size=128):

    '''Aligned and cropped CelebA data downloaded from 
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'''

    img_transforms = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                                 std = (0.5, 0.5, 0.5))
            ])

    celeba_dataset = ImageDataset(images_dir, image_size, img_transforms)

    return celeba_dataset
