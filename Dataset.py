from PIL import Image
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path)


class Dataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None, loader=default_loader):
        self.data_frame=pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        img_name=self.data_frame.iloc[index,1]
        label=self.data_frame.iloc[index,2:].fillna(0)
        label=np.array([label])
        label=label.astype('float').reshape(-1,1,1)
        img = self.loader(img_name)
        if self.transform is not None:
            img = self.transform(img)

        return img,label,img_name




