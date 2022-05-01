import torch.nn as nn
import torch.nn.functional as F
import torch


class VargoNet(nn.Module):

    def __init__(self):
        super(VargoNet, self).__init__()

        ##Combinada
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 9, stride=2, padding=0 ) 
        self.batchNorm1 = nn.BatchNorm2d(num_features=96), 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 4)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1) 
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
        self.fc1  = nn.Linear(in_features= 3456, out_features= 2304)
        self.fc2  = nn.Linear(in_features= 2304, out_features= 4096)
        self.fc3  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc4 = nn.Linear(in_features=4096 , out_features=102)

        


    def forward(self,x):

        x = F.relu(self.conv1(x)) # out_dim [110x110x96]
        x = self.maxpool(x) # out_dim [55x55x96]
        x = F.relu(self.conv2(x))  # out_dim [29x29x256]
        x = self.maxpool(x)  # out_dim [14x14x256]
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
