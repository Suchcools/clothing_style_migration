import numpy
import numpy as np
import pandas as pd
import os 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class CycleGANDataset(Dataset):
    def __init__(self):
        self.path='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/JPEGImages/'
        self.maskPath='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug/'
        self.images_list = [x for x in os.listdir(self.path) if 'dataset10k' in x][:100]
        self.labels_list = [x for x in os.listdir('dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug') if 'dataset10k' in x][:100]

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,index):
        img=cv2.imread(self.path+self.images_list[index])
        mask = np.array(cv2.imread(self.maskPath+self.labels_list[index],0))
        mask = np.where(np.logical_and(mask >= 4, mask <= 8), 1,0) # 把非衣服部分挖空
        return normalization(img.transpose(2,0,1)),normalization(cv2.stylization(img, sigma_s=60, sigma_r=0.6).transpose(2,0,1)),mask

dataset = CycleGANDataset()  # 自己定义的数据集
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

import transforms
import Tuple

def get_fundamental_transforms(
    inp_size: Tuple[int, int], pixel_mean: np.array, pixel_std: np.array
) -> transforms.Compose:
    """
    Returns the core transforms needed to feed the images to our model (refer notebook for the 4 operations).

    Args:
    - inp_size: tuple (height, width) denoting the dimensions for input to the model
    - pixel_mean: the mean of the raw dataset [Shape=(1,)]
    - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
    Returns:
    - fundamental_transforms: transforms.Compose with the 3 fundamental transforms
    """
    return transforms.Compose(
        [
            transforms.Resize(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std)
        ]
    )



from sklearn.model_selection import train_test_split

class CycleGANDataset(Dataset):
    def __init__(self):
        self.path='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/JPEGImages/' #图片路径
        self.maskPath='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug/' #服装蒙版路径
        self.images_list = [x for x in os.listdir(self.path) if 'dataset10k' in x][:100]
        self.labels_list = [x for x in os.listdir('dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug') if 'dataset10k' in x][:100]
        
        # 随机化排序
        self.images_list, self.labels_list = np.shuffle(self.images_list, self.labels_list)

        # 划分训练集和测试集
        self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(self.images_list, self.labels_list, test_size=0.2, random_state=42)

    def __len__(self):
        return len(self.images_train)

    def __getitem__(self,index):
        img=cv2.imread(self.path+self.images_train[index])
        mask = np.array(cv2.imread(self.maskPath+self.labels_train[index],0))
        mask = np.where(np.logical_and(mask >= 4, mask <= 8), 1,0) # 把非衣服部分挖空
        return normalization(img.transpose(2,0,1)),normalization(cv2.stylization(img, sigma_s=60, sigma_r=0.6).transpose(2,0,1)),mask