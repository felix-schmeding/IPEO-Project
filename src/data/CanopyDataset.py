from torch.utils.data import Dataset

from skimage.io import imread

import os
import glob
import torchvision.transforms as T
import torch
import pandas as pd

# ! to change
mean=torch.tensor([0.504, 0.504, 0.503])
std=torch.tensor([0.019 , 0.018, 0.018])

# normalize image [0-1] (or 0-255) to zero-mean unit standard deviation
normalize = T.Normalize(mean, std)
# we invert normalization for plotting later
std_inv = 1 / (std + 1e-7)
unnormalize = T.Normalize(-mean * std_inv, std_inv)

default_transform =  T.Compose([
        T.ToTensor()
        #normalize])
        ])

class CanopyDataset(Dataset):

    def __init__(self, dataset_root='data/', split_file='data/data_split.csv', transforms=default_transform, split='train'):
        self.transforms = transforms

        # read split file
        split_df = pd.read_csv(split_file)
        label_folder = 'labels/'
        split_df = (split_df
                        .drop(index=0)      # image 0 doesn't exist
                        .loc[split_df['split'] == split, :]    # only the split wanted
                        ) 
        split_df['label_path']  =  split_df['label_path'].map(lambda x : label_folder + x.split('/')[1])  

        self.data = []        # list of tuples of (image path, label path)

        for _, row in split_df.iterrows():    # df with all images of the split
            # img_path = os.path.join(dataset_root, row['image_path'])
            # label_path = os.path.join(dataset_root, row['label_path'])
            img_path = dataset_root + row['image_path']
            label_path = dataset_root + row['label_path']

            self.data.append((
                img_path,
                label_path
            ))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, x):
        imgName, labelName = self.data[x]
        img = imread(imgName)   # numpy arry (36,36,12)
        label = imread(labelName)

        # * only if we need it, throws errors with image format
        if self.transforms is not None: # only apply to image, not label
            img = self.transforms(img)
        
        return img, label
