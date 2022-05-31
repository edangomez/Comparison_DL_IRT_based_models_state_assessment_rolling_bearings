#! /usr/bin/env python

# Tutorial for creating your own dataloader function

# Link to download sample data: https://download.pytorch.org/tutorial/hymenoptera_data.zip
import os
#from skimage import io, transform
import cv2 as cv
from numpy import dtype

import torch
from torch.utils.data import Dataset
from util.func import csv2numpy
import numpy as np


# Dataset class

class IRT(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images_paths, labels, data_path, dtype = 'img', transform=None):
        super(IRT, self).__init__()
        'Initialization'
        self.labels = labels
        self.images_paths = images_paths
        self.transform = transform
        self.data_path = data_path
        self.dtype = dtype

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_paths)

  def __getitem__(self, index):
        if self.dtype == 'irt':
            'Generates one sample of data'
            #   print(self.images_paths[index])
            path = self.images_paths[index]
            
            # Load data and get label
            label = self.labels[index]
            image = csv2numpy(path)
            shape1 = image.shape
            # print(shape1)
            if image.shape == (120, 161):
                  image = cv.resize(image, (321,240), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
            # shape2 = image.shape
            # if shape1!= shape2
            # print(image.shape)
            image = np.array([image])
            # image= np.array([cv.resize(image, (321,240), interpolation=cv.INTER_CUBIC)])  #W,H,C   [200, 300, 3]
            # image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

            if self.transform:
                  image = self.transform(image)
            return image, label
        elif self.dtype == 'img':
            'Generates one sample of data'
            #   print(self.images_paths[index])
            path = self.images_paths[index]
            
            # Load data and get label
            label = self.labels[index]
            image = cv.imread(path, cv.IMREAD_COLOR)
            shape1 = image.shape
            # print(image.shape)
            if image.shape == (640, 480, 3):
                  image = np.rot90(image)
            shape2 = image.shape
            # if shape1!= shape2:
            #       breakpoint()
            image= cv.resize(image, (640,480), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
            image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

            if self.transform:
                  image = self.transform(image)
            return image, label


class IRTHybrid(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images_paths, labels, data_path, dtype = 'img', transform=None):
        super(IRTHybrid, self).__init__()
        'Initialization'
        self.labels = labels
        self.images_paths = images_paths[:,0]
        self.irt_paths = images_paths[:,1]
        self.transform = transform
        self.data_path = data_path
        self.dtype = dtype

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_paths)

  def __getitem__(self, index):
      'Generates one sample of data'
      ## IRT DATA
      #   print(self.images_paths[index])
      path_irt = self.irt_paths[index]
      # print(path_irt)
      # Load data and get label
      label = self.labels[index]
      irt = csv2numpy(path_irt)
      # print(irt.shape)
      irt= np.array([cv.resize(irt, (227,227), interpolation=cv.INTER_CUBIC)])  #W,H,C   [200, 300, 3]
      # image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

      ## IMG DATA

      'Generates one sample of data'
      #   print(self.images_paths[index])
      path = self.images_paths[index]
      
      # Load data and get label
      label = self.labels[index]
      image = cv.imread(path, cv.IMREAD_COLOR)
      #   print(image.shape)
      image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
      image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

      if self.transform:
            image = self.transform(image)
            irt = self.transform(irt)
      return image, irt, label

class IRTHybrid2(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images_paths, labels, data_path, dtype = 'img', transform=None):
        super(IRTHybrid2, self).__init__()
        'Initialization'
        self.labels = labels
        self.images_paths = images_paths[:,0]
        self.irt_paths = images_paths[:,1]
        self.transform = transform
        self.data_path = data_path
        self.dtype = dtype

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_paths)

  def __getitem__(self, index):
      'Generates one sample of data'
      ## IRT DATA
      #   print(self.images_paths[index])
      path_irt = self.irt_paths[index]
      # print(path_irt)
      # Load data and get label
      label = self.labels[index]
      irt = csv2numpy(path_irt)
      if irt.shape == (120, 161):
            irt = cv.resize(irt, (321,240), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
      # shape2 = image.shape
      # if shape1!= shape2
      # print(image.shape)
      irt = np.pad(irt, ((120,120),(159, 160)))
      # irt = cv.resize(irt, (640,480), interpolation=cv.INTER_CUBIC)
      irt = np.array([irt])
      # print(irt.shape)
      # irt= np.array([cv.resize(irt, (227,227), interpolation=cv.INTER_CUBIC)])  #W,H,C   [200, 300, 3]
      # image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

      ## IMG DATA

      'Generates one sample of data'
      #   print(self.images_paths[index])
      path = self.images_paths[index]
      
      # Load data and get label
      label = self.labels[index]
      image = cv.imread(path, cv.IMREAD_COLOR)
      shape1 = image.shape
            # print(image.shape)
      if image.shape == (640, 480, 3):
            image = np.rot90(image)
      shape2 = image.shape
      # if shape1!= shape2:
      #       breakpoint()
      image= cv.resize(image, (640,480), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
      #   print(image.shape)
      # image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
      image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

      if self.transform:
            image = self.transform(image)
            irt = self.transform(irt)
      return image, irt, label



# class IRT(Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, images_paths, labels, data_path, transform=None):
#         super(IRT, self).__init__()
#         'Initialization'
#         self.labels = labels
#         self.images_paths = images_paths
#         self.transform = transform
#         self.data_path = data_path

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.images_paths)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         path = self.images_paths[index]
#       #   print(path)
#         # Load data and get label
#         label = self.labels[index]
#         image = cv.imread(path, cv.IMREAD_COLOR)
#         image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
#         image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

#         if self.transform:
#             image = self.transform(image)
#         return image, label
    

# class IRT(Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, images_paths, labels, data_path, transform=None, dtype = 'img'):
#         super(IRT, self).__init__()
#         'Initialization'
#         self.labels = labels
#         self.images_paths = images_paths
#         self.transform = transform
#         self.data_path = data_path
#         self.dtype = dtype
#         self.labelsdic = {0: np.array([1,0,0]), 1: np.array([0,1,0]), 3:np.array([0,0,1])}
#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.images_paths)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         if self.dtype == 'img':
#             path = self.images_paths[index]
#             # Load data and get label
#             label = self.labels[index]
#             image = cv.imread(path, cv.IMREAD_COLOR)
#             # print(path, image.shape)
#             image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
#             # image= cv.resize(image, (224,224), interpolation=cv.INTER_CUBIC)  #W,H,C   [200, 300, 3]
#             # print(path, image.shape)
#             #image = image.transpose(2, 0, 1) #C,W,H  [3,200,300]

#             if self.transform:
#                   image = self.transform(image)
#             return image, label
#         elif self.dtype == 'irt':
#             path = self.images_paths[index]
#             # Load data and get label
#             label = self.labels[index]

#             image = csv2numpy(path)
#             # image= cv.resize(image, (224,224), interpolation=cv.INTER_CUBIC) 
#             image= cv.resize(image, (227,227), interpolation=cv.INTER_CUBIC) 

#             if self.transform:
#                   image = self.transform(image)
#             return image, label