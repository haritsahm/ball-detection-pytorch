import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCNN_v1(nn.Module):
  def __init__(self):
    super(FCNN_v1, self).__init__()
    # Input size [3, 150, 200]
    
    self.conv1_1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=2)
    self.conv1_2 = nn.Conv2d(3, 16, kernel_size=9, stride=2, padding=3)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=(3,2))
    self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
    self.fc_x_1 = nn.Linear(32*38*50, 100)
    self.fc_x_2 = nn.Linear(100, 200)
    self.fc_y_1 = nn.Linear(32*38*50, 100)
    self.fc_y_2 = nn.Linear(100, 150)

    self.drop_out = nn.Dropout2d(p=0.5)

    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

  def forward(self, x):
    out_1 = self.drop_out(F.leaky_relu(self.conv1_1(x))) 
    out_2 = self.drop_out(F.leaky_relu(self.conv1_2(x)))
    out = torch.cat((out_1, out_2),1)
    out = self.drop_out(F.leaky_relu(self.conv2(out)))
    out = self.drop_out(F.leaky_relu(self.conv3(out)))
    print(out.shape)
    out = torch.flatten(out, 1)
    out_x = self.drop_out(F.leaky_relu(self.fc_x_1(out)))
    out_x = self.fc_x_2(out_x)
    out_y = self.drop_out(F.leaky_relu(self.fc_y_1(out)))
    out_y = self.fc_y_2(out_y)

    return out_x, out_y

  class FCNN_v2(nn.Module):
  def __init__(self, input_size = (150,200)):
    super(FCNN_v2, self).__init__()
    # Input size [3, 150, 200]
    self._input_size = np.asarray(input_size) if len(input_size) == 2 else np.asarray(input_size[1:])

    ## Encoding
    self.e_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
    self.e_bn1 = nn.BatchNorm2d(16)

    self.e_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
    self.e_bn2 = nn.BatchNorm2d(32)

    self.e_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
    self.e_bn3 = nn.BatchNorm2d(32)

    self.e_conv4 = nn.Conv2d(48, 64, kernel_size=3, padding="same")
    self.e_bn4 = nn.BatchNorm2d(64)

    self.e_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
    self.e_bn5 = nn.BatchNorm2d(64)

    self.e_conv6 = nn.Conv2d(112, 128, kernel_size=3, padding="same")
    self.e_bn6 = nn.BatchNorm2d(128)

    self.e_conv7 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
    self.e_bn7 = nn.BatchNorm2d(128)

    ## Decoding
    self.d_conv1 = nn.Conv2d(176, 64, kernel_size=3, padding="same")
    self.d_bn1 = nn.BatchNorm2d(64)

    self.d_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding="same")
    self.d_bn2 = nn.BatchNorm2d(32)

    self.d_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
    self.d_bn3 = nn.BatchNorm2d(32)

    self.d_conv4 = nn.Conv2d(48, 16, kernel_size=3, padding="same")
    self.d_bn4 = nn.BatchNorm2d(16)

    self.d_conv5 = nn.Conv2d(16, 16, kernel_size=3, padding="same")
    self.d_bn5 = nn.BatchNorm2d(16)

    self.d_conv6 = nn.Conv2d(16, 1, kernel_size=3, padding="same")

    self.drop_out = nn.Dropout2d(p=0.5)

    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

  def forward(self, x):
    out = self.drop_out(F.leaky_relu(self.e_bn1(self.e_conv1(x))))
    bn_out = out
    out = F.max_pool2d(out, 2, 2)
    maxpool_1 = out

    out = self.drop_out(F.leaky_relu(self.e_bn2(self.e_conv2(out))))
    out = self.drop_out(F.leaky_relu(self.e_bn3(self.e_conv3(out))))

    out = torch.cat((out, maxpool_1),1)
    cat_out = out
    out = F.max_pool2d(out, 2, 2)
    maxpool_2 = out

    out = self.drop_out(F.leaky_relu(self.e_bn4(self.e_conv4(out))))
    out = self.drop_out(F.leaky_relu(self.e_bn5(self.e_conv5(out))))

    out = torch.cat((out, maxpool_2),1)
    out = self.drop_out(F.leaky_relu(self.e_bn6(self.e_conv6(out))))
    out = self.drop_out(F.leaky_relu(self.e_bn7(self.e_conv7(out))))
    out = F.interpolate(out, size=tuple((self._input_size/2).astype(np.uint8)), mode="bilinear", align_corners=True)
    out = torch.cat((out, cat_out),1)

    out = self.drop_out(F.leaky_relu(self.d_bn1(self.d_conv1(out))))
    out = self.drop_out(F.leaky_relu(self.d_bn2(self.d_conv2(out))))

    out = self.drop_out(F.leaky_relu(self.d_bn3(self.d_conv3(out))))
    out = F.interpolate(out, tuple(self._input_size.astype(np.uint8)), mode="bilinear", align_corners=True)
    out = torch.concat((out, bn_out), 1)

    out = self.drop_out(F.leaky_relu(self.d_bn4(self.d_conv4(out))))
    out = self.drop_out(F.leaky_relu(self.d_bn5(self.d_conv5(out))))
    out = self.d_conv6(out)

    return out