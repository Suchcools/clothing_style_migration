import numpy
import numpy as np
import pandas as pd
import os 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class CycleGANDataset(Dataset):
    def __init__(self):
        self.path='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/JPEGImages/' #图片路径
        self.maskPath='dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug/' #服装蒙版路径
        self.images_list = [x for x in os.listdir(self.path) if 'dataset10k' in x][:100]
        self.labels_list = [x for x in os.listdir('dataset/ICCV15_fashion_dataset(ATR)/humanparsing/SegmentationClassAug') if 'dataset10k' in x][:100]

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,index):
        img=cv2.imread(self.path+self.images_list[index])
        mask = np.array(cv2.imread(self.maskPath+self.labels_list[index],0))
        mask = np.where(np.logical_and(mask >= 4, mask <= 8), 1,0) # 把非衣服部分挖空
        return normalization(img.transpose(2,0,1)),normalization(cv2.stylization(img, sigma_s=60, sigma_r=0.6).transpose(2,0,1)),mask
    
def Mask_replace(Real_img,Fake_img,Mask):

    # 确定掩码区域的边界框
    mask_indices = np.where(Mask == 1)
    y1, x1 = np.min(mask_indices, axis=1)
    y2, x2 = np.max(mask_indices, axis=1)

    # 将B中的掩码区域提取出来
    B_masked = Fake_img[y1:y2, x1:x2]
    B_masked = cv2.resize(B_masked, (x2-x1, y2-y1))  # 将B的掩码区域缩放到与A的掩码区域相同大小

    # 将B的掩码区域复制到A的对应区域
    A_masked = Real_img[y1:y2, x1:x2]
    A_masked[Mask[y1:y2, x1:x2] == 1] = B_masked[Mask[y1:y2, x1:x2] == 1]

    # 生成新的图片矩阵
    result = Real_img.copy()
    result[y1:y2, x1:x2] = A_masked
    return result