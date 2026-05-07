import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    """
    Simple multilayer perceptron for MNIST experiments.
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class TinyCNN(nn.Module):
    """
    Small CNN for CIFAR-10 experiments.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class TinyResidualBlock(nn.Module):
    """
    Minimal residual block for testing.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        out += residual
        out = F.relu(out)

        return out


class TinyResNet(nn.Module):
    """
    Lightweight ResNet-style model for optimization experiments.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.block1 = TinyResidualBlock(32)
        self.block2 = TinyResidualBlock(32)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))

        x = self.block1(x)
        x = self.block2(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = TinyMLP()

    sample = torch.randn(8, 1, 28, 28)

    output = model(sample)

    print("Output shape:", output.shape)
