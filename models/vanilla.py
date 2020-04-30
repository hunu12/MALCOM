import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        return self.fc(out)
