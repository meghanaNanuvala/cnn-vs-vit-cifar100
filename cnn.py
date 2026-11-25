"""The CNN: ResNet-18 for 32x32 images, trained from scratch on CIFAR-100.

    python cnn.py      trains the model and writes results/cnn.json
"""

import torch
import torch.nn as nn

import train


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection around them."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # If the block changes the resolution or the number of channels, the
        # input cannot be added to the output as it is. A 1x1 convolution on
        # the skip path brings it to the same shape.
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        skip = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + skip)


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # The ImageNet ResNet starts with a 7x7 stride-2 convolution and a
        # max-pool, which would shrink a 32x32 image to 8x8 straight away.
        # For CIFAR the stem is a single 3x3 convolution at full resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self.make_stage(64, 64, stride=1)      # 32x32
        self.stage2 = self.make_stage(64, 128, stride=2)     # 16x16
        self.stage3 = self.make_stage(128, 256, stride=2)    # 8x8
        self.stage4 = self.make_stage(256, 512, stride=2)    # 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1),
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


if __name__ == "__main__":
    args = train.setup()
    # channels_last is the faster memory layout for convolutions under mixed precision
    model = ResNet18(train.NUM_CLASSES).to(train.device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    train.run("cnn", "ResNet-18", model, optimizer, args)
