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

import albumentations as albu

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



"""
    class for generating the frame ids, helper function of class::VSLDataSet
"""
# reference: https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
class FramesSampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1): 
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

        
"""
    class for Dataset
    @ param:
        1. image_paths: r'./dataset/train' (the dir of dataset)
        
        2. seq_length: the video clip of training, default as 10
        
        3. transform_list: a list of transform for data augmentation: 
                eg. [albu.Compose([...]), albu.Compose([...])], 
                first transform is resize to (3, 224, 224) for all the images, second is blur & rgb transform & ...
                
        4. imba_transform_dict: a dict of transform for data augmentation, to perform different transform for each class.
                eg. { 'BOO': albu.Compose([...]), 'OOL': albu.Compose([...]) }, default as None, which means there is no imbalanced class
                
        5. length: number of all the video clips
"""
class VSLDataSet(Dataset):
    def __init__(self, image_paths, seq_length, transform_list, imba_transform_dict, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform_list = transform_list
        self.imba_transform_dict = imba_transform_dict
        self.imba_ids = []
        if self.imba_transform_dict != None:
            self.imba_ids = self.imba_transform_dict.keys()
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        indices = list(range(start, end))
        images = []
        label = self.image_paths[start][1]
        y = torch.tensor(label, dtype=torch.long)
        if((label in self.imba_ids) and (self.imba_transform_dict != None)): #if we set imba transform，and current example's label is the key
            for i in indices:
                image_path = self.image_paths[i][0]
                image = np.array(Image.open(image_path))
                images.append(image)
            target = {}
            for i, image in enumerate(images[1:]):
                target['image' + str(i)] = 'image'
            augmented = albu.Compose(self.imba_transform_dict[label], p=1, additional_targets=target)(image=images[0],
                                                                        image0=images[1],
                                                                        image1=images[2],
                                                                        image2=images[3],
                                                                        image3=images[4],
                                                                        image4=images[5],
                                                                        image5=images[6],
                                                                        image6=images[7],
                                                                        image7=images[8],
                                                                        image8=images[9])
            images_ = []
            for img_name in augmented.keys():
                print(img_name)
                img = augmented[img_name]
                img = Image.fromarray(img)
                images_.append(self.transform_list[0](img))
            x = torch.stack(images_)

        else: #for all the data we apply the same transform
            if(self.transform_list[1] == None):
                images = []
                for i in indices:
                    image_path = self.image_paths[i][0]
                    image = Image.open(image_path)
                    image = self.transform_list[0](image)
                    images.append(image)
                x = torch.stack(images)

            else:
                for i in indices:
                    image_path = self.image_paths[i][0]
                    image = np.array(Image.open(image_path))
                    images.append(image)
                target = {}
                for i, image in enumerate(images[1:]):
                    target['image' + str(i)] = 'image'
                augmented = albu.Compose(self.transform_list[1], p=1, additional_targets=target)(image=images[0],
                                                                        image0=images[1],
                                                                        image1=images[2],
                                                                        image2=images[3],
                                                                        image3=images[4],
                                                                        image4=images[5],
                                                                        image5=images[6],
                                                                        image6=images[7],
                                                                        image7=images[8],
                                                                        image8=images[9])
                images_after_transform = []
                for img_name in augmented.keys():
                    img_after_transform = augmented[img_name]
                    img_after_transform = Image.fromarray(img_after_transform)
                    images_after_transform.append(self.transform_list[0](img_after_transform))
                x = torch.stack(images_after_transform)
        return x, y
    
    def __len__(self):
        return self.length

        
"""
    function to generate dataset
    @ param:
        1. root_dir: the dir of a dataset(train/test/valid): eg. '~/Thesis/code/dataset/train/'
        
        2. class_name_to_id: the class_name and index's table
        
        3. aug_transform_list: a list of transform for data augmentation: 
                eg. [albu.Compose([...]), albu.Compose([...])], 
                first transform is resize to (3, 224, 224) for all the images, second is blur & rgb transform & ...
                
        4. imba_transform_dict: a dict of transform for data augmentation, to perform different transform for each class.
                eg. { 'BOO': albu.Compose([...]), 'OOL': albu.Compose([...]) }, default as None, which means there is no imbalanced class
"""
def load_dataset(root_dir, class_name_to_id, aug_transform_list, imba_transform_dict = None):

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
    seq_length = 10 #TODO: 16

    sampler = FramesSampler(end_idx, seq_length)


    dataset = VSLDataSet(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform_list=aug_transform_list,
        imba_transform_dict = imba_transform_dict,
        length=len(sampler))

    return dataset, sampler

    
"""
    function to generate dataloader for train/valid/test set, here you can define the transform_list and data_root_dir
    @ param:
        1. train_batch_size: the size of number of video clips of one batch
        
        2. valid_batch_size: the same for valid set
        
        3. valid_batch_size: the same for test set
"""    
def create_dataloader_train_valid_test(train_batch_size=32, valid_batch_size=16, test_batch_size=16):
    
    common_transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
    ])
    

    train_transform = albu.Compose([
        albu.RandomBrightnessContrast(p=0.5),
        albu.Blur(p=0.5),
        albu.HueSaturationValue(p=0.5),
        albu.ShiftScaleRotate(scale_limit=(-0.1, 0),rotate_limit=20, border_mode=0),
        albu.RGBShift(p=0.5),
        albu.CLAHE(p=0.5),
        albu.RandomGamma(p=0.5)
    ])


    train_dataset, train_sampler = load_dataset(root_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/train/', class_name_to_id = class_name_to_id_, aug_transform_list = [common_transform_tensor, train_transform])
    
    valid_dataset, valid_sampler = load_dataset(root_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/valid/', class_name_to_id = class_name_to_id_, aug_transform_list = [common_transform_tensor, None])

    test_dataset, test_sampler = load_dataset(root_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/test/', class_name_to_id = class_name_to_id_, aug_transform_list = [common_transform_tensor, None])
    
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory = True
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        sampler=valid_sampler,
        num_workers=8,
        pin_memory = True
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=8,
        pin_memory = True
    )        
    
    dataloaders = {
    'train': train_data_loader,
    'valid': valid_data_loader,
    'test': test_data_loader,
    }
    return dataloaders



"""
    a example to use the dataloader
"""    
if __name__ == '__main__':

    dataloaders = create_dataloader_train_valid_test(train_batch_size=4, valid_batch_size=1, test_batch_size=1)
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']
    
    num_epochs = 3
    valid_epoch_step = 1
    test_epoch_step = 2
    import matplotlib.pyplot as plt
    for epoch in range(num_epochs):
        # training
        print('Train:')
        for index, (data, target) in enumerate(train_dataloader):
        #    print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
        
            print(data.shape)
            fig=plt.figure(figsize=(12, 6))
            fig.add_subplot(2,3,1)
            plt.imshow(data[0,0].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(2,3,2)
            plt.imshow(data[0,1].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(2,3,3)
            plt.imshow(data[0,2].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(2,3,4)
            plt.imshow(data[0,-3].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(2,3,5)
            plt.imshow(data[0,-2].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(2,3,6)
            plt.imshow(data[0,-1].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            plt.savefig('foo.png')
            input('test')
        
        # validation
        # model.eval()
        if (epoch%valid_epoch_step == 0):
            valid_losses = []
            print('Valid:')
            for index, (data, target) in enumerate(valid_dataloader):
                print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)

                
        # test
        # model.eval()
        if (epoch%test_epoch_step == 0):
            valid_losses = []
            print('Test:')
            for index, batch in enumerate(test_dataloader):
                print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)

        
    
