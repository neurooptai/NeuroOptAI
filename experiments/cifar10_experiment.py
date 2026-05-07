import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuro_opt.controllers.cost_aware_early_stopping import (
    CostAwareEarlyStopping,
)

from neuro_opt.controllers.adaptive_lr_controller import (
    AdaptiveLRController,
)

from neuro_opt.controllers.meta_controller import (
    MetaController,
)


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    model = SimpleCNN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    early_stopper = CostAwareEarlyStopping(
        patience=5,
        min_gain=1e-3,
        cost_weight=1e-5,
    )

    lr_controller = AdaptiveLRController()

    meta_controller = MetaController(
        stop_controller=early_stopper,
        lr_controller=lr_controller,
    )

    epochs = 30

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        decision = meta_controller.decide(
            loss=avg_loss,
            optimizer=optimizer,
            compute_cost=len(train_loader),
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if decision["stop"]:

            print("Training stopped by MetaController.")

            break

    print("Experiment completed.")


if __name__ == "__main__":
    main()
