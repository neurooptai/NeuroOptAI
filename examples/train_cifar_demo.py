import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neuro_opt.controllers.cost_aware_early_stopping import (
    CostAwareEarlyStopping
)

from neuro_opt.controllers.adaptive_lr_controller import (
    AdaptiveLRController
)

from neuro_opt.controllers.meta_controller import (
    MetaController
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# --------------------------------------------------
# CIFAR-10 Dataset
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# --------------------------------------------------
# Simple CNN Model
# --------------------------------------------------

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


model = SimpleCNN().to(device)


# --------------------------------------------------
# Loss + Optimizer
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# NeuroOptAI Controllers
# --------------------------------------------------

stop_controller = CostAwareEarlyStopping(
    patience=5,
    min_gain=1e-3,
    cost_weight=1e-5
)

lr_controller = AdaptiveLRController(
    factor_down=0.5,
    factor_up=1.02,
    patience=3
)

meta_controller = MetaController(
    stop_controller,
    lr_controller
)


# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            predicted = torch.argmax(pred, dim=1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    accuracy = evaluate()

    decision = meta_controller.decide(
        loss=avg_loss,
        optimizer=optimizer,
        compute_cost=len(train_loader)
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    if decision["stop"]:

        print("Training stopped by NeuroOptAI controller.")

        break


# --------------------------------------------------
# Save Model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "cifar10_neuroopt_model.pth"
)

print("Model saved.")
