import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from scipy.stats import kurtosis, skew, entropy, median_absolute_deviation
import cv2

## Modified AlexNet implementation ------------------------------------------------------------------------------------------------------------------------------------------------

class AlexNet(nn.Module):
    
    def __init__(self, dtype, resize_type):
        super(AlexNet, self).__init__()
        self.dtype = dtype
        self.resize_type = resize_type
        ##Combinada
        if self.dtype == 'img':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        elif dtype == 'hybrid':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        if dtype == 'hybrid2':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456+5, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        elif dtype == 'irt':
            print('using irt_mat data')
            self.conv1 = nn.Conv2d(in_channels=1, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)

    def forward(self,img):

        x = F.relu(self.conv1(img)) # out_dim [110x110x96]
        x = self.maxpool(x) # out_dim [55x55x96]
        x = F.relu(self.conv2(x))  # out_dim [29x29x256]
        x = self.maxpool(x)  # out_dim [14x14x256]
        # plt.imshow(self.weight2[0, 2])
        # plt.show()
        x = F.relu(self.conv3(x)) # out_dim [14x14x384]
        x = self.maxpool(x)  # out_dim [7x7x256]
        x = F.relu(self.conv4(x)) # out_dim [8x8x384]
        x = self.maxpool(x)  # out_dim [4x4x256]
        x = self.maxpool(x)  # out_dim [6x6x256]
        x = x.reshape(x.shape[0], -1)  # out_dim [9216x1]
        x = F.relu(self.fc1(x)) # out_dim [4096x1]
        x = F.relu(self.fc2(x)) # out_dim [4096x1]
        x = F.relu(self.fc3(x)) # out_dim [4096x1]
        x = self.fc4(x) # out_dim [1000x1]

        return x
class HyAlexNet(nn.Module):
    def __init__(self, dtype, resize_type):
        super(HyAlexNet, self).__init__()
        self.dtype = dtype
        self.resize_type = resize_type
        ##Combinada
        if self.dtype == 'img':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        elif self.dtype == 'hybrid':
            self.conv1 = nn.Conv2d(in_channels=4, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        elif self.dtype == 'hybrid2':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.weight2 = self.conv2.weight.data.numpy()

            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 26885, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)
        elif self.dtype == 'irt':
            print('using irt_mat data')
            self.conv1 = nn.Conv2d(in_channels=1, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
            self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
            self.fc1  = nn.Linear(in_features= 5760, out_features= 2304)
            self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
            self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
            self.fc4 = nn.Linear(in_features=4096 , out_features=4)

    def forward(self,img, irt=None, gpuID = 1):
        # print(img.size, irt.size)
        if self.dtype in ['img', 'irt']:
            x = F.relu(self.conv1(img)) # out_dim [110x110x96]
            x = self.maxpool(x) # out_dim [55x55x96]
            x = F.relu(self.conv2(x))  # out_dim [29x29x256]
            x = self.maxpool(x)  # out_dim [14x14x256]
            # plt.imshow(self.weight2[0, 2])
            # plt.show()
            x = F.relu(self.conv3(x)) # out_dim [14x14x384]
            x = self.maxpool(x)  # out_dim [7x7x256]
            x = F.relu(self.conv4(x)) # out_dim [8x8x384]
            x = self.maxpool(x)  # out_dim [4x4x256]
            x = self.maxpool(x)  # out_dim [6x6x256]
            x = x.reshape(x.shape[0], -1)  # out_dim [9216x1]
            # print(x.shape)
            x = F.relu(self.fc1(x)) # out_dim [4096x1]
            x = F.relu(self.fc2(x)) # out_dim [4096x1]
            x = F.relu(self.fc3(x)) # out_dim [4096x1]
            x = self.fc4(x) # out_dim [1000x1]
        elif self.dtype == 'hybrid':
            img = torch.cat((img, irt), axis = 1)
            x = F.relu(self.conv1(img)) # out_dim [110x110x96]
            x = self.maxpool(x) # out_dim [55x55x96]
            x = F.relu(self.conv2(x))  # out_dim [29x29x256]
            x = self.maxpool(x)  # out_dim [14x14x256]
            # plt.imshow(self.weight2[0, 2])
            # plt.show()
            x = F.relu(self.conv3(x)) # out_dim [14x14x384]
            x = self.maxpool(x)  # out_dim [7x7x256]
            x = F.relu(self.conv4(x)) # out_dim [8x8x384]
            x = self.maxpool(x)  # out_dim [4x4x256]
            x = self.maxpool(x)  # out_dim [6x6x256]
            x = x.reshape(x.shape[0], -1)  # out_dim [9216x1]
            x = F.relu(self.fc1(x)) # out_dim [4096x1]
            x = F.relu(self.fc2(x)) # out_dim [4096x1]
            x = F.relu(self.fc3(x)) # out_dim [4096x1]
            x = self.fc4(x) # out_dim [1000x1]
        elif self.dtype == 'hybrid2':
            if self.resize_type == 'partial':
                print('partial')
                print(type(img), img.shape)
                img = img.cpu().numpy()
                print('after: ', type(img), img.shape)
                img= cv2.resize(img, (227,227), interpolation=cv2.INTER_CUBIC)  #W,H,C   [200, 300, 3]\
                img = torch.from_numpy(img).cuda(gpuID)
            if self.resize_type in ['original', 'partial']:
                # print('features from original thermal_mat')
                irt = irt[:,:,120:-120,159:-160].cpu().numpy()
                # print(0 in np.unique(irt))
            else:
                irt = irt.cpu().numpy()
                # print(irt.shape)
            # print(x.shape, irt.shape)
            # print(np.std(irt, axis = (2,3)).shape)
            # print(irt.min(axis = (2,3)).shape)
            feat_vect = np.concatenate((np.mean(irt, axis = (2,3)), np.median(irt, axis = (2,3)), np.std(irt, axis = (2,3)), irt.max(axis = (2,3)), irt.min(axis = (2,3))),axis =1)
            feat_vect = torch.from_numpy(feat_vect).cuda(gpuID)
            
            x = F.relu(self.conv1(img)) # out_dim [110x110x96]
            x = self.maxpool(x) # out_dim [55x55x96]
            x = F.relu(self.conv2(x))  # out_dim [29x29x256]
            x = self.maxpool(x)  # out_dim [14x14x256]
            # plt.imshow(self.weight2[0, 2])
            # plt.show()
            x = F.relu(self.conv3(x)) # out_dim [14x14x384]
            x = self.maxpool(x)  # out_dim [7x7x256]
            x = F.relu(self.conv4(x)) # out_dim [8x8x384]
            x = self.maxpool(x)  # out_dim [4x4x256]
            x = self.maxpool(x)  # out_dim [6x6x256]
            x = x.reshape(x.shape[0], -1)  # out_dim [9216x1]
            # print(x.shape)
            x = torch.cat((x, feat_vect), axis=1)
            # print(x.shape)
            x = F.relu(self.fc1(x)) # out_dim [4096x1]
            x = F.relu(self.fc2(x)) # out_dim [4096x1]
            x = F.relu(self.fc3(x)) # out_dim [4096x1]
            x = self.fc4(x) # out_dim [1000x1]

        return x

        
## EfficientNet implementation ----------------------------------------------------------------------------------------------------------------------------------------------------

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups, # for depth-wise convolution --> if groups=1 is a normal convolution, but if groups=in_channels is a depth-wise convolution
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1), # restore dimensionality
            nn.Sigmoid(), # asigns value [0, 1] as attention scores --> importance
        )

    def forward(self, x):
        return x * self.se(x) # each channel is multiplied by the attention scores

class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4, # squeeze excitation
            survival_prob=0.8, # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1 # if we downsample we cannot do residual connection (HxW won't match)
        hidden_dim = in_channels * expand_ratio 
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs # If we do have residuals, can perform stochastic depth
        else:
            return self.conv(x) # otherwise, just normal MBCONV


class EfficientNet(nn.Module):
    def __init__(self,  dtype, version, num_classes, resize_type):
        super(EfficientNet, self).__init__()
        self.dtype = dtype
        self.resize_type = resize_type
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels, dtype) # most of the backbone here
        if self.dtype in ['hybrid2', 'hybrid3']:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(last_channels+5, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(last_channels, num_classes),
            )

    def calculate_factors(self, version, alpha=1.2, beta=1.1): # default values from paper 
        phi, res, drop_rate = phi_values[version] # we are not going to use resolution coefficient
        depth_factor = alpha ** phi # how many should L_i from paper increase
        width_factor = beta ** phi # how manny should W_i from paper increase
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels, dtype):
        channels = int(32 * width_factor)
        if dtype in ['img',  'hybrid2']:
            features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        elif dtype == 'irt':
            features = [CNNBlock(1, channels, 3, stride=2, padding=1)]
        elif dtype in ['hybrid', 'hybrid3']:
            features = [CNNBlock(4, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, img, irt=None, gpuID = 1):
        if self.dtype in ['img', 'irt']:
            x = img
            x = self.pool(self.features(x))
            return self.classifier(x.view(x.shape[0], -1))
        elif self.dtype == 'hybrid':
            # print(irt.shape, img.shape)
            x = torch.cat((img, irt), axis = 1)
            x = self.pool(self.features(x))
            return self.classifier(x.view(x.shape[0], -1))
        elif self.dtype == 'hybrid2':
            # print('irt dim: ', irt.shape)
            # print('img dim: ', img.shape)
            if self.resize_type == 'partial':
                print('partial')
                print(type(img), img.shape)
                img = img.cpu().numpy()
                print('after: ', type(img), img.shape)
                img= cv2.resize(img, (227,227), interpolation=cv2.INTER_CUBIC)  #W,H,C   [200, 300, 3]\
                img = torch.from_numpy(img).cuda(gpuID)
            x = img
            x = self.pool(self.features(x))
            x = x.view(x.shape[0], -1)
            
            if self.resize_type in ['original', 'partial']:
                # print('features from original thermal_mat')
                irt = irt[:,:,120:-120,159:-160].cpu().numpy()
                # print(0 in np.unique(irt))
            else:
                irt = irt.cpu().numpy()
                # print(irt.shape)
            # print(x.shape, irt.shape)
            # print(np.std(irt, axis = (2,3)).shape)
            # print(irt.min(axis = (2,3)).shape)
            feat_vect = np.concatenate((np.mean(irt, axis = (2,3)), np.median(irt, axis = (2,3)), np.std(irt, axis = (2,3)), irt.max(axis = (2,3)), irt.min(axis = (2,3))),axis =1)
            feat_vect = torch.from_numpy(feat_vect).cuda(gpuID)
            # print(x.shape, feat_vect.shape)
            x = torch.cat((x, feat_vect), axis=1)
            return self.classifier(x)
        elif self.dtype == 'hybrid3':
            # print('irt dim: ', irt.shape)
            # print('img dim: ', img.shape)
            if self.resize_type == 'partial':
                print('partial')
                print(type(img), img.shape)
                img = img.cpu().numpy()
                print('after: ', type(img), img.shape)
                img= cv2.resize(img, (227,227), interpolation=cv2.INTER_CUBIC)  #W,H,C   [200, 300, 3]\
                img = torch.from_numpy(img).cuda(gpuID)
            x = torch.cat((img, irt), axis = 1)
            x = self.pool(self.features(x))
            x = x.view(x.shape[0], -1)
            
            if self.resize_type in ['original', 'partial']:
                # print('features from original thermal_mat')
                irt = irt[:,:,120:-120,159:-160].cpu().numpy()
                # print(0 in np.unique(irt))
            else:
                irt = irt.cpu().numpy()
                # print(irt.shape)
            # print(x.shape, irt.shape)
            # print(np.std(irt, axis = (2,3)).shape)
            # print(irt.min(axis = (2,3)).shape)
            feat_vect = np.concatenate((np.mean(irt, axis = (2,3)), np.median(irt, axis = (2,3)), np.std(irt, axis = (2,3)), irt.max(axis = (2,3)), irt.min(axis = (2,3))),axis =1)
            feat_vect = torch.from_numpy(feat_vect).cuda(gpuID)
            # print(x.shape, feat_vect.shape)
            x = torch.cat((x, feat_vect), axis=1)
            return self.classifier(x)
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)

    print(model(x).shape) # (num_examples, num_classes)


## VGG Implementation -----------------------------------------------------------------------------------------------------------------------------------

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG_net(nn.Module):
    def __init__(self, dtype, resize_type, num_classes=4):
        super(VGG_net, self).__init__()
        self.dtype =dtype
        if dtype in ['img', 'hybrid2']:
            self.in_channels = 3
        elif dtype == 'irt':
            self.in_channels = 1
        elif dtype in ['hybrid', 'hybrid3']:
            self.in_channels = 4
        self.conv_layers = self.create_conv_layers(VGG_types["VGG19"])
        self.resize_type = resize_type

        if self.dtype in ['hybrid2', 'hybrid3']:
            self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7+5#+128512
            , 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))
        else:
            self.fcs = nn.Sequential(
            nn.Linear(25088#512 * 7 * 7+10752#+128512
            , 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))

    def forward(self, img, irt =None, gpuID = 1):
        if self.dtype in ['irt', 'img']:
            x = self.conv_layers(img)
            x = x.reshape(x.shape[0], -1)
            x = self.fcs(x)
        elif self.dtype == 'hybrid':
            x = torch.cat((img, irt), axis = 1)
            x = self.conv_layers(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fcs(x)
        elif self.dtype == 'hybrid2':
            # print(img.shape)
            x = self.conv_layers(img)
            if self.resize_type in ['original', 'partial']:
                # print('features from original thermal_mat')
                irt = irt[:,:,120:-120,159:-160].cpu().numpy()
                # print(0 in np.unique(irt))
            else:
                irt = irt.cpu().numpy()
                # print(irt.shape)
            # print(x.shape, irt.shape)
            # print(np.std(irt, axis = (2,3)).shape)
            # print(irt.min(axis = (2,3)).shape)
            feat_vect = np.concatenate((np.mean(irt, axis = (2,3)), np.median(irt, axis = (2,3)), np.std(irt, axis = (2,3)), irt.max(axis = (2,3)), irt.min(axis = (2,3))),axis =1)
            feat_vect = torch.from_numpy(feat_vect).cuda(gpuID)
            x = x.reshape(x.shape[0], -1)
            x = torch.cat((x, feat_vect), axis=1)
            x = self.fcs(x)
        elif self.dtype == 'hybrid3':
            # print(img.shape, irt.shape)
            x = torch.cat((img, irt), axis = 1)
            x = self.conv_layers(x)
            if self.resize_type in ['original', 'partial']:
                # print('features from original thermal_mat')
                irt = irt[:,:,120:-120,159:-160].cpu().numpy()
                # print(0 in np.unique(irt))
            else:
                irt = irt.cpu().numpy()
                # print(irt.shape)
            # print(x.shape, irt.shape)
            # print(np.std(irt, axis = (2,3)).shape)
            # print(irt.min(axis = (2,3)).shape)
            feat_vect = np.concatenate((np.mean(irt, axis = (2,3)), np.median(irt, axis = (2,3)), np.std(irt, axis = (2,3)), irt.max(axis = (2,3)), irt.min(axis = (2,3))),axis =1)
            feat_vect = torch.from_numpy(feat_vect).cuda(gpuID)
            x = x.reshape(x.shape[0], -1)
            x = torch.cat((x, feat_vect), axis=1)
            x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


