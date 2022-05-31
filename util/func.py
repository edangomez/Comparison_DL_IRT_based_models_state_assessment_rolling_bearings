import os
import numpy as np
import shutil
import glob
import csv
import cv2
import matplotlib.pyplot as plt
from genericpath import exists
from typing import Sized
from matplotlib.figure import Figure
import json
from PIL import Image
import pandas as pd
from skimage import color
from skimage import io
from scipy.signal import correlate
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.stats import kurtosis, skew, entropy, median_absolute_deviation
import warnings
import scipy.optimize


def csv2numpy(route):
    f = open(route, "r", encoding="utf16")
    f = f.readlines()[5:] #create a list where thermal pixel matrix begins

    # delete the index from each row and convert it into a list
    for i in range(len(f)):
        f[i] = list(f[i])[f[i].index(',')+1:-3]#[3:-3])

    # converts each row of strings into a list of integers
    for j in range(len(f)):
        # x corresponde a particiones temporales que inician con la ubicación de la primera coma

        i = -1

        # se inicializa con toda la fila actual
        x = f[j][i+1:]

        # list that will contain all floats from partition x
        t = []

        # While the partition x contains more than 2 possible floats
        while len(x)>11:
            x = x[i+1:] # x is a partition of itself stating in the location of th first ','

            coma = x.index(',') # index of the first coma in the partition
            t.append(float(''.join(x[0:coma]))) # creates a float of the elements of x until the first coma\
                                                # which are the temperature data

            i = coma # asigns the first coma index to i so that the new x partition is created in the next cicle

        t.append(float(''.join(x[0:coma]))) # the remaining possible floats are added to t
        t.append(float(''.join(x[coma+1:])))

        f[j] = t # the current row of the archive is replaced by the list of floats
    f = np.array(f) # the archive is converted to a numpy array
    return f

def feat_vect(img):
    '''
    Returns feature vector of a given image or array of temperatures
    :param img: 2D np.ndarray
    :return: 1D np.ndarray feature vector
    '''
    i = img.flatten()
    vec = [i.mean(), i.std(), kurtosis(i), skew(i), entropy(i), median_absolute_deviation(i), max(i), min(i)]
    return vec

def feat_matroi(train_list, r_start, r_end, c_start, c_end):
    feat_v = np.zeros([len(train_list), 8])
    for i in range(len(train_list)):
        # Read csv a convert to 2D numpy array
        img = csv2numpy(train_list[i])
        roi = (img[r_start: r_end, c_start: c_end] - 32) * 5 / 9

        # feature vector to feature matrix
        vect = feat_vect(roi)
        feat_v[i] += vect

    return feat_v  # , np.array([rois])