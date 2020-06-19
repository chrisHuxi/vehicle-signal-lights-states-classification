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
        #print('Getting images from {} to {}'.format(start, end))
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

def load_dataset(root_dir, class_name_to_id, aug_transform):

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

    end_idx = [0, *end_idx]
    
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    seq_length = 10

    sampler = FramesSampler(end_idx, seq_length)


    dataset = VSLDataSet(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=aug_transform,
        length=len(sampler))

    return dataset, sampler

def create_dataloader_train_valid_test(train_batch_size=32, valid_batch_size=16, test_batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset, train_sampler = load_dataset(root_dir = '../dataset/train/', class_name_to_id = class_name_to_id_, aug_transform = train_transform)
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    valid_dataset, valid_sampler = load_dataset(root_dir = '../dataset/valid/', class_name_to_id = class_name_to_id_, aug_transform = valid_transform)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset, test_sampler = load_dataset(root_dir = '../dataset/test/', class_name_to_id = class_name_to_id_, aug_transform = test_transform)
    
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        sampler=valid_sampler
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler
    )        
    
    dataloaders = {
    'train': train_data_loader,
    'valid': valid_data_loader,
    'test': test_data_loader,
    }
    return dataloaders
if __name__ == '__main__':

    dataloaders = create_dataloader_train_valid_test()
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']
    
    num_epochs = 3
    valid_epoch_step = 2
    test_epoch_step = 2
    for epoch in range(num_epochs):
        # training
        # model.train()
        train_losses = []
        print('Train:')
        for index, (data, target) in enumerate(train_dataloader):

            print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
            #loss = self._train_step(epoch, index, batch)
            #losses.append(loss)

        # validation
        # model.eval()
        
        if (epoch%valid_epoch_step == 0):
            valid_losses = []
            print('Valid:')
            for index, batch in enumerate(valid_dataloader):
                
                print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
                #loss = self._val_step(epoch, index, batch)
                #losses.append(loss)
                
        # validation
        # model.eval()
        if (epoch%test_epoch_step == 0):
            valid_losses = []
            print('Test:')
            for index, batch in enumerate(test_dataloader):
                
                print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
                #loss = self._val_step(epoch, index, batch)
                #losses.append(loss)
        
    