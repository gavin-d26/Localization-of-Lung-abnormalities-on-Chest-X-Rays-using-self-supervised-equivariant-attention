import torch
import torchvision
import torchvision.transforms as tr
from torchvision.transforms.transforms import Normalize, ToTensor
import torchvision.io as io
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import PIL


torch.manual_seed(0)
random.seed(0)

        
# returns train and validation DataFrames        
def load_test_validation_df(path, val_split_index):
    train_splits = []
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        df = pd.read_csv(file_path)
        train_splits.append(df) 

    val_splits = []    
    for i in val_split_index:
        val_splits.append(train_splits.pop(i))

    train_df = pd.concat(train_splits, ignore_index = True) 
    val_df = pd.concat(val_splits, ignore_index = True) 
    train_df = train_df.drop(['Unnamed: 0'], axis = 1)
    val_df = val_df.drop(['Unnamed: 0'], axis = 1)
    
    print('train_split_len: ', sum, len(train_df)) 
    print('val_split_len: ', sum, len(val_df))

    return train_df, val_df        


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, df, validation = False):
        super(CXRDataset, self).__init__()
        self.size = 256
        self.df = df
        self.df['Finding Labels'] = self.df['Finding Labels'].apply(lambda x: [int(i) for i in x.split()])
        self.length = len(df)
        if validation is False:
            self.transforms = tr.Compose([
                                        tr.ToTensor(),
                                        tr.RandomResizedCrop(self.size, scale = (0.80,1.0)),
                                        tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        tr.RandomHorizontalFlip(),
                                        tr.RandomRotation(15),
                                        ])
        else:
            self.transforms = tr.Compose([tr.ToTensor(),
                                        tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
                
        
    def __getitem__(self, index):
        label, img_path = self.df.loc[index, 'Finding Labels'], self.df.loc[index, 'path']
        label = torch.tensor(label).float()
        label = label[:-1]
        image = PIL.Image.open(img_path).resize((self.size, self.size), resample= PIL.Image.BILINEAR)
        image = image.convert('RGB')
        image = self.transforms(image)
        return image, label
        
        
    def __len__(self):
        return self.length        

        