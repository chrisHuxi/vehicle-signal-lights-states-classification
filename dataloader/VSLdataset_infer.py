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



# reference: https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
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

class VSLDataSet(Dataset):
    def __init__(self, image_paths, seq_length, transform_list, length):
        '''
        constructor of VSLDataLoader
        @ param:
            1. image_paths: r'./dataset/train' (the dir of dataset)
            2. transform: image_transform (for augmentation of image)
            4. class_name_to_id: dict convert one of class name to id
        '''
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform_list = transform_list
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []

        if(self.transform_list[1] == None):
            images = []
            for i in indices:
                image_path = self.image_paths[i]
                image = Image.open(image_path)
                image = self.transform_list[0](image)
                images.append(image)
            x = torch.stack(images)

        else:
            for i in indices:
                image_path = self.image_paths[i]
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
        return x, self.image_paths[end].split('/')[-1]
    
    def __len__(self):
        return self.length


def alphanum_key(s):
    return int((s.split('/')[-1]).split('.png')[0])


def load_dataset_infer(root_dir, aug_transform_list, track_folder_name):

    track_image_paths = {}
    list_temp = []
    end_idx = []
    for d in os.scandir(root_dir):
        track_id = d.name
        track_image_paths[track_id] = []
        if d.is_dir: #every track
            paths = glob.glob(os.path.join(d.path, '*.png'))
            paths.sort(key = alphanum_key)
            track_image_paths[track_id].extend(paths)
            end_idx.extend([len(paths)])

    end_idx = [0, *end_idx]

    end_idx = torch.cumsum(torch.tensor(end_idx), 0)

    seq_length = 25 #TODO: 16


    dataset = VSLDataSet(
        image_paths=track_image_paths[track_folder_name],
        seq_length=seq_length,
        transform_list=aug_transform_list,
        length=len(track_image_paths[track_folder_name]) - 1)

    return dataset

def create_dataloader_infer(infer_batch_size, track_folder_name):
    
    common_transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    

    infer_dataset= load_dataset_infer(root_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/inference_data/', aug_transform_list = [common_transform_tensor, None], track_folder_name = track_folder_name)


    infer_data_loader = DataLoader(
        infer_dataset,
        batch_size=infer_batch_size,
        num_workers=8,
        pin_memory = True
    )
    
    dataloaders = {
    'infer': infer_data_loader,
    }
    return dataloaders


if __name__ == '__main__':

    dataloaders = create_dataloader_infer(infer_batch_size=1)
    infer_dataloader = dataloaders['infer']
    
    num_epochs = 1
    import matplotlib.pyplot as plt
    for epoch in range(num_epochs):
        # training
        print('Train:')
        for index, (data, target) in enumerate(infer_dataloader):

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
            plt.close()
    
