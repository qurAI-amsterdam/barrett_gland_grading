from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from einops import rearrange
import math


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        """
        Paramaters for Deconvolution were chosen to avoid artifacts, following
        link https://distill.pub/2016/deconv-checkerboard/
        """
        super(DecoderBlockV2, self).__init__()

        self.in_channels = in_channels
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class DenseUnet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with DenseNet-161 encoder

    """

    def __init__(self, num_classes=3, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with DenseNet-161
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.densenet169(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0,
                                   self.encoder.features.relu0,
                                   self.encoder.features.pool0)

        self.conv2 = self.encoder.features.denseblock1
        self.transition1 = self.encoder.features.transition1

        self.conv3 = self.encoder.features.denseblock2
        self.transition2 = self.encoder.features.transition2

        self.conv4 = self.encoder.features.denseblock3
        self.transition3 = self.encoder.features.transition3

        self.conv5 = self.encoder.features.denseblock4

        self.center = DecoderBlockV2(1664, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(1664 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1280 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec_seg = ConvRelu(num_filters, num_filters)
        self.final_seg = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        transition1 = self.transition1(conv2)
        conv3 = self.conv3(transition1)
        transition2 = self.transition2(conv3)
        conv4 = self.conv4(transition2)
        transition3 = self.transition3(conv4)
        conv5 = self.conv5(transition3)
        
        # classification
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec_seg = self.dec_seg(dec1)

        # Segmentation
        x_out = self.final_seg(dec_seg)

        return x_out
    
    def get_embeddings(self):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        transition1 = self.transition1(conv2)
        conv3 = self.conv3(transition1)
        transition2 = self.transition2(conv3)
        conv4 = self.conv4(transition2)
        transition3 = self.transition3(conv4)
        conv5 = self.conv5(transition3)
        
        # classification
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        
        return dec1 
        

