import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import models
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import numpy as np

from benchpush.baselines.ship_ice_nav.planning_based.utils.networks.network_utils import conv_out_size, conv_trans_out_size, maxpool_size

class ResLayer(nn.Module):
    def __init__(self, num_features, hidden_features):
        super(ResLayer, self).__init__()

        self.linear1 = nn.Linear(num_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, num_features)

    def forward(self, x_input):
        x = nn.functional.relu(self.linear1(x_input))
        x = nn.functional.relu(self.linear2(x))
        x = x + x_input
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, middle_stride, input_H, input_W):
        super(BottleNeck, self).__init__()

        self.in_channes = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.input_H = input_H
        self.input_W = input_W
        self.middle_stride = middle_stride

        # 1st conv2d kernel = 1, stride = 1, padding = 0
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1_H = conv_out_size(inp_size=self.input_H, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1_W = conv_out_size(inp_size=self.input_W, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn_1 = nn.BatchNorm2d(num_features=hidden_channels)

        # 2nd conv2d kernel = 3, stride = (1 or 2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=middle_stride, padding=1, dilation=1)
        self.conv2_H = conv_out_size(inp_size=self.conv1_H, kernel_size=3, stride=middle_stride, padding=1, dilation=1)
        self.conv2_W = conv_out_size(inp_size=self.conv1_W, kernel_size=3, stride=middle_stride, padding=1, dilation=1)
        self.bn_2 = nn.BatchNorm2d(num_features=hidden_channels)

        # 3rd conv2d kernel = 1, stride = 1, padding = 0
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3_H = conv_out_size(inp_size=self.conv2_H, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3_W = conv_out_size(inp_size=self.conv2_W, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)

        if middle_stride == 2:
            self.conv_residule = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, dilation=1)
            self.residule_H = conv_out_size(inp_size=self.input_H, kernel_size=1, stride=2, padding=0, dilation=1)
            self.residule_W = conv_out_size(inp_size=self.input_W, kernel_size=1, stride=2, padding=0, dilation=1)
            self.bn_residule = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.conv_residule = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
            self.residule_H = conv_out_size(inp_size=self.input_H, kernel_size=1, stride=1, padding=0, dilation=1)
            self.residule_W = conv_out_size(inp_size=self.input_W, kernel_size=1, stride=1, padding=0, dilation=1)
            self.bn_residule = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x_input):

        # go through convolution stem
        x = self.bn_1(self.conv1(x_input))
        x = self.bn_2(self.conv2(x))
        x = nn.functional.relu(self.bn_3(self.conv3(x)))

        # add residule
        res = self.bn_residule(self.conv_residule(x_input))
        x = x + res
        return x



class UNet_Ice(nn.Module):
    def __init__(self, input_includes_centroids=False, output_includes_centroids=False):
        super(UNet_Ice, self).__init__()
        
        # encoder kernels
        down_kernel_size = 3

        # decoder kernels
        up_kernnel_size = 2

        self.output_includes_centroids = output_includes_centroids

        self.input_H = 40
        self.input_W = 40
        # self.input_H = 32
        # self.input_W = 32

        # encoder stage 1 ----------------------------------------
        if input_includes_centroids:
            self.conv_down_1_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        else:
            self.conv_down_1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_1_1_H = conv_out_size(inp_size=self.input_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_1_1_W = conv_out_size(inp_size=self.input_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.conv_down_1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_down_1_2_H = conv_out_size(inp_size=self.conv_down_1_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_1_2_W = conv_out_size(inp_size=self.conv_down_1_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_down_1 = nn.BatchNorm2d(num_features=32)

        self.conv_down_1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_down_1_3_H = conv_out_size(inp_size=self.conv_down_1_2_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_1_3_W = conv_out_size(inp_size=self.conv_down_1_2_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)


        # encoder stage 2 ---------------------------------------
        self.conv_down_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_2_1_H = conv_out_size(inp_size=self.conv_down_1_3_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_2_1_W = conv_out_size(inp_size=self.conv_down_1_3_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.conv_down_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_down_2_2_H = conv_out_size(inp_size=self.conv_down_2_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_2_2_W = conv_out_size(inp_size=self.conv_down_2_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_down_2 = nn.BatchNorm2d(num_features=64)

        self.conv_down_2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1, bias=False)
        self.conv_down_2_3_H = conv_out_size(inp_size=self.conv_down_2_2_H, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1)
        self.conv_down_2_3_W = conv_out_size(inp_size=self.conv_down_2_2_W, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1)

        # encoder stage 3 ----------------------------
        self.conv_down_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_3_1_H = conv_out_size(inp_size=self.conv_down_2_3_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_3_1_W = conv_out_size(inp_size=self.conv_down_2_3_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.conv_down_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_down_3_2_H = conv_out_size(inp_size=self.conv_down_3_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_down_3_2_W = conv_out_size(inp_size=self.conv_down_3_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_down_3 = nn.BatchNorm2d(num_features=128)

        self.conv_down_3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1, bias=False)
        self.conv_down_3_3_H = conv_out_size(inp_size=self.conv_down_3_2_H, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1)
        self.conv_down_3_3_W = conv_out_size(inp_size=self.conv_down_3_2_W, kernel_size=down_kernel_size, stride=2, padding=1, dilation=1)

        # bottle neck ------------------------------------
        self.bottleNeck1 = BottleNeck(in_channels=128, out_channels=256, hidden_channels=64, middle_stride=1, 
                                      input_H=self.conv_down_3_3_H, input_W=self.conv_down_3_3_W)
        
        self.bottleNeck2 = BottleNeck(in_channels=256, out_channels=256, hidden_channels=64, middle_stride=1, 
                                      input_H=self.bottleNeck1.conv3_H, input_W=self.bottleNeck1.conv3_W)

        # decoder stage 3 ---------------------------
        self.conv_up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, output_padding=0)
        self.conv_up_3_H = conv_trans_out_size(inp_size=self.bottleNeck2.conv3_H, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, out_padding=0)
        self.conv_up_3_W = conv_trans_out_size(inp_size=self.bottleNeck2.conv3_W, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, out_padding=0)

        self.decode_3_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_3_1_H = conv_out_size(inp_size=self.conv_up_3_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_3_1_W = conv_out_size(inp_size=self.conv_up_3_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.decode_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.decode_3_2_H = conv_out_size(inp_size=self.decode_3_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_3_2_W = conv_out_size(inp_size=self.decode_3_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_up_3 = nn.BatchNorm2d(num_features=128)


        # decoder stage 2 -------------------------------
        self.conv_up_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, output_padding=0)
        self.conv_up_2_H = conv_trans_out_size(inp_size=self.decode_3_2_H, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, out_padding=0)
        self.conv_up_2_W = conv_trans_out_size(inp_size=self.decode_3_2_W, kernel_size=up_kernnel_size, stride=2, padding=0, dilation=1, out_padding=0)

        self.decode_2_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_2_1_H = conv_out_size(inp_size=self.conv_up_2_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_2_1_W = conv_out_size(inp_size=self.conv_up_2_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.decode_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.decode_2_2_H = conv_out_size(inp_size=self.decode_2_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_2_2_W = conv_out_size(inp_size=self.decode_2_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_up_2 = nn.BatchNorm2d(num_features=64)


        # decoder stage 1 -----------------------------
        self.decode_1_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_1_1_H = conv_out_size(inp_size=self.decode_2_2_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_1_1_W = conv_out_size(inp_size=self.decode_2_2_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.decode_1_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.decode_1_2_H = conv_out_size(inp_size=self.decode_1_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_1_2_W = conv_out_size(inp_size=self.decode_1_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.decode_1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.decode_1_3_H = conv_out_size(inp_size=self.decode_1_1_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.decode_1_3_W = conv_out_size(inp_size=self.decode_1_1_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

        self.bn_up_1 = nn.BatchNorm2d(num_features=32)

        # output layer
        if output_includes_centroids:
            self.conv_final_centroids = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.conv_final_H = conv_out_size(inp_size=self.decode_1_3_H, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)
        self.conv_final_W = conv_out_size(inp_size=self.decode_1_3_W, kernel_size=down_kernel_size, stride=1, padding=1, dilation=1)

    
    def skip_connection(self, x_encode, x_decode):
        return torch.cat((x_encode, x_decode), dim=1)


    def forward(self, x):

       # encoder stage 1 ---------------------------------------
        x = nn.functional.relu(self.bn_down_1(self.conv_down_1_1(x)))
        x_encode_1 = nn.functional.relu(self.bn_down_1(self.conv_down_1_2(x)))
        x = nn.functional.relu(self.bn_down_1(self.conv_down_1_3(x_encode_1)))

        # encoder stage 2 --------------------------------------
        x = nn.functional.relu(self.bn_down_2(self.conv_down_2_1(x)))
        x_encode_2 = nn.functional.relu(self.bn_down_2(self.conv_down_2_2(x)))
        x = nn.functional.relu(self.bn_down_2(self.conv_down_2_3(x_encode_2)))

        # encoder stage 3 --------------------------------------
        x = nn.functional.relu(self.bn_down_3(self.conv_down_3_1(x)))
        x_encode_3 = nn.functional.relu(self.bn_down_3(self.conv_down_3_2(x)))
        x = nn.functional.relu(self.bn_down_3(self.conv_down_3_3(x_encode_3)))

        # bottle neck ------------------------------------------
        x = self.bottleNeck1(x)
        x = self.bottleNeck2(x)

        # decoder stage 3 --------------------------------------
        x = self.conv_up_3(x)
        x = self.skip_connection(x_encode_3, x)
        x = nn.functional.relu(self.bn_up_3(self.decode_3_1(x)))
        x = nn.functional.relu(self.bn_up_3(self.decode_3_2(x)))

        # decoder stage 2 --------------------------------------
        x = self.conv_up_2(x)
        x = self.skip_connection(x_encode_2, x)
        x = nn.functional.relu(self.bn_up_2(self.decode_2_1(x)))
        x = nn.functional.relu(self.bn_up_2(self.decode_2_2(x)))

        # decoder stage 1 --------------------------------------
        x = nn.functional.relu(self.bn_up_1(self.decode_1_1(x)))
        x = self.skip_connection(x_encode_1, x)
        x = nn.functional.relu(self.bn_up_1(self.decode_1_2(x)))
        x = nn.functional.relu(self.bn_up_1(self.decode_1_3(x)))

        # final stage
        occ = self.conv_final(x)
        occ = nn.functional.sigmoid(occ)

        if self.output_includes_centroids:
            centroids = self.conv_final_centroids(x)
            return occ, centroids
        else:
            return occ
        