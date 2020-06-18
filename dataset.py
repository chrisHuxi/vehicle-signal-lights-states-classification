import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import img_to_tensor
import torchvision.transforms as transforms
import glob
import json
import os
import PIL.Image
import PIL.ImageDraw
from PIL import Image

# for image augmentation
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

# http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal#labelDef
'''
label defination
OOO: Brake light and turn signals off
BOO: Brake light on, turn signals off
OLO: Brake light off, left signal on
BLO: Brake light on, left signal on
OOR: Brake light off, right signal on
BOR: Brake light on, right signal on
OLR: Brake light off, left and right signal on (hazard warning light on)
BLR: Brake light on, left and right signal on (hazard warning light on)
'''

class_name_to_id_ = {
'OOO':0,
'BOO':1,
'OLO':2,
'BLO':3,
'OOR':4,
'BOR':5,
'OLR':6,
'BLR':7
}

def image_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=100, min_width=100, p=0.5),
        RandomCrop(height=512, width=960, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        #RandomAffine(30)
    ], p=p)

# reference: https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
class FramesSampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1): #滑动窗口
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)


class VSLDataSet(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        '''
        constructor of VSLDataLoader
        @ param:
            1. image_paths: r'./dataset/train' (the dir of dataset)
            2. transform: image_transform (for augmentation of image)
            3. mode: 'train', 'valid' or 'test' (test mode will not return label)
            4. class_name_to_id: dict convert one of class name to id
        '''
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        
        return x, y
    
    def __len__(self):
        return self.length


    
if __name__ == '__main__':
    root_dir = './dataset/train/'
    class_name_to_id = {
    'OOO':0,
    'BOO':1,
    'OLO':2,
    'BLO':3,
    'OOR':4,
    'BOR':5,
    'OLR':6,
    'BLR':7
    }
    #class_paths = class_name_to_id_.keys()
    class_image_paths = []
    end_idx = []

    for class_name in (class_name_to_id_.keys()):
        class_idx = class_name_to_id[class_name]
        class_path = os.path.join(root_dir, class_name)
        for d in os.scandir(class_path):
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                # Add class idx to paths
                paths = [(p, class_idx) for p in paths]
                class_image_paths.extend(paths)
                end_idx.extend([len(paths)])
    print(end_idx)
    end_idx = [0, *end_idx]
    
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    print(end_idx)
    seq_length = 10

    sampler = FramesSampler(end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = VSLDataSet(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=3,
        sampler=sampler
    )

    for data, target in loader:
        print(data.shape)
        print(target.shape)
