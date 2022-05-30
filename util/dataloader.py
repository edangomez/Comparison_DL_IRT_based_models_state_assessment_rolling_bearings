#! /usr/bin/env python

# Tutorial for creating your own dataloader function

# Link to download sample data: https://download.pytorch.org/tutorial/hymenoptera_data.zip
import os
#from skimage import io, transform
import cv2 as cv

import torch
from torch.utils.data import Dataset


class IRT(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images_paths, labels, data_path, transform=None):
        super(IRT, self).__init__()
        'Initialization'
        self.labels = labels
        self.images_paths = images_paths
        self.transform = transform
        self.data_path = data_path

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        path = self.images_paths[index]
        # Load data and get label
        label = self.labels[index]
        image = cv.imread(path, cv.IMREAD_COLOR)
        image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
        #image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

        if self.transform:
            image = self.transform(image)
        return image, label
    