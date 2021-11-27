import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models import resnet

class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(self,gradient,w,h,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, shared_bias=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.locationBias=shared_bias if shared_bias is not None else torch.nn.Parameter(torch.zeros((w,h,3)))
        self.locationEncode=torch.autograd.Variable(torch.ones(w,h,3))
        if gradient:
            for i in range(w):
                self.locationEncode[i,:,1]=self.locationEncode[:,i,0]=i/float(w-1)
        
        self.up=torch.nn.Upsample(size=(w,h), mode='bilinear', align_corners=False)
        self.w=w
        self.h=h
    def forward(self,inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias=self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())
        b=self.locationBias*self.locationEncode
        convRes=super().forward(inputs)
        if convRes.shape[2]!=self.w and convRes.shape[3]!=self.h:
            convRes=self.up(convRes)
        return convRes+b[:,:,0]+b[:,:,1]+b[:,:,2]

class NimbroNetV2(nn.Module):
  def __init__(self, base='resnet18', shared_bias = True, pretrained=True, num_classes=3, input_size=(224,224)):
    super(NimbroNetV2, self).__init__()

    self._input_size = np.asarray(input_size) if len(input_size) == 2 else np.asarray(input_size[1:])

    self.enc = resnet.resnet18(pretrained=pretrained, num_classes=num_classes, features_only=True)
    self.pwconv_1 = nn.Conv2d(64, 128, kernel_size=1)
    self.pwconv_2 = nn.Conv2d(128, 256, kernel_size=1)
    self.pwconv_3 = nn.Conv2d(256, 256, kernel_size=1)

    self.dec_convt_1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
    self.dec_bn_1 = nn.BatchNorm2d(512)

    self.dec_convt_2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
    self.dec_bn_2 = nn.BatchNorm2d(512)

    self.dec_convt_3 = nn.ConvTranspose2d(512, 128, 2, stride=2)
    self.dec_bn_3 = nn.BatchNorm2d(256)
    if shared_bias:
      self.shared_bias = torch.nn.Parameter(torch.zeros((int(self._input_size[0]/4), int(self._input_size[1]/4),3)))
    else:
      self.shared_bias = None
    self.seg_head = LocationAwareConv2d(True,int(self._input_size[1]/4),int(self._input_size[1]/4), in_channels=256, out_channels=3, kernel_size=1, padding=0, shared_bias=self.shared_bias)
    self.det_head = LocationAwareConv2d(True,int(self._input_size[1]/4),int(self._input_size[1]/4), in_channels=256, out_channels=3, kernel_size=1, padding=0, shared_bias=self.shared_bias)

  def forward(self, x):
    # Encoder
    x = self.enc.conv1(x)
    x = self.enc.bn1(x)
    x = self.enc.act1(x)
    x = self.enc.maxpool(x)

    x = self.enc.layer1(x)
    layer_1_pw = self.pwconv_1(x)
    x = self.enc.layer2(x)
    layer_2_pw = self.pwconv_2(x)
    x = self.enc.layer3(x)
    layer_3_pw = self.pwconv_3(x)
    x = self.enc.layer4(x)

    # Decoder 
    x = self.dec_convt_1(F.relu(x))
    x = torch.cat((x, layer_3_pw), 1)
    x = F.relu(self.dec_bn_1(x))
    x = self.dec_convt_2(x)
    x = torch.cat((x, layer_2_pw), 1)
    x = F.relu(self.dec_bn_2(x))
    x = self.dec_convt_3(x)
    x = torch.cat((x, layer_1_pw), 1)
    x = F.relu(self.dec_bn_3(x))
    seg_out = self.seg_head(x)
    det_out = self.det_head(x)

    return seg_out, det_out
