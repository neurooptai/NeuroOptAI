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


# =========================
# Simple Neural Network
# =========================

class TinyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


# =========================
# Training Function
# =========================

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # -------------------------
    # Dataset
    # -------------------------

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # -------------------------
    # Model
    # -------------------------

    model = TinyNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    # -------------------------
    # NeuroOptAI Controllers
    # -------------------------

    early_stopper = CostAwareEarlyStopping(
        patience=5,
        min_gain=1e-3,
        cost_weight=1e-5
    )

    lr_controller = AdaptiveLRController(
        factor_down=0.5,
        factor_up=1.02
    )

    meta_controller = MetaController(
        stop_controller=early_stopper,
        lr_controller=lr_controller
    )

    # -------------------------
    # Training Loop
    # -------------------------

    epochs = 20

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            predictions = model(x)

            loss = loss_fn(predictions, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -------------------------
        # NeuroOptAI Decision
        # -------------------------

        decision = meta_controller.decide(
            loss=avg_loss,
            optimizer=optimizer,
            compute_cost=len(train_loader)
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.6f} "
            f"LR: {current_lr:.8f}"
        )

        print(f"Decision: {decision}")

        # -------------------------
        # Stop if controller decides
        # -------------------------

        if decision["stop"]:

            print("\nNeuroOptAI stopped training.")
            print("Optimization no longer economically useful.")

            break

    print("\nTraining complete.")


# =========================
# Main
# =========================

if __name__ == "__main__":

    train()
