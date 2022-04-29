import os
from re import L
import time
import math
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# class Trainer(object):

#     def __init__(self, model, generator, optimizerG, trainDataloader, valDataloader,\
#          gpuID, nBatch = 10, out = 'train', maxEpoch =1, cuda=True, lrDecayEpoch={}) -> None:
        