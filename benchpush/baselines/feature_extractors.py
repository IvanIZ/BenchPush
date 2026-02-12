import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class ResNet18(BaseFeaturesExtractor):
    """
    A custom feature extractor based on ResNet 18
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, weights=None):
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]

        # load resnet-18 model
        self.resnet = resnet18(weights=weights)

        # modify the first conv layer to accept channels according to the environment
        self.resnet.conv1 = nn.Conv2d(
            in_channels=n_input_channels,  # Set the new input channels
            out_channels=self.resnet.conv1.out_channels,  # Keep the same number of output channels
            kernel_size=self.resnet.conv1.kernel_size,  # Keep kernel size
            stride=self.resnet.conv1.stride,  # Keep stride
            padding=self.resnet.conv1.padding,  # Keep padding
            bias=self.resnet.conv1.bias is not None,  # Match bias usage
        )

        # remove the last fully connected layer, which was designed for classification
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return x

####################################################################################################
# For Spatial Action Maps

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_input_channels=3, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    #nn.init.constant_(m.bn2.weight, 0)
                    pass

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18_SAM(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

class DenseActionSpaceDQN(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet18_SAM(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(32, num_output_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)