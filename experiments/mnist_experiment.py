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
# MODEL
# --------------------------------------------------

class SimpleMNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")

    # -----------------------------
    # DATASET
    # -----------------------------

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = SimpleMNISTNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # CONTROLLERS
    # -----------------------------

    stopper = CostAwareEarlyStopping(
        patience=5,
        min_gain=1e-3,
        cost_weight=1e-5
    )

    lr_controller = AdaptiveLRController(
        factor_down=0.5,
        factor_up=1.02
    )

    controller = MetaController(
        stop_controller=stopper,
        lr_controller=lr_controller
    )

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------

    epochs = 25

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            predictions = model(x)

            loss = loss_fn(predictions, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predicted = predictions.argmax(dim=1)

            correct += (predicted == y).sum().item()

            total += y.size(0)

        avg_loss = total_loss / len(train_loader)

        train_accuracy = 100 * correct / total

        # -----------------------------
        # META DECISION
        # -----------------------------

        decision = controller.decide(
            loss=avg_loss,
            optimizer=optimizer,
            compute_cost=len(train_loader)
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # -----------------------------
        # VALIDATION
        # -----------------------------

        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for x, y in test_loader:

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                predicted = outputs.argmax(dim=1)

                test_correct += (predicted == y).sum().item()

                test_total += y.size(0)

        test_accuracy = 100 * test_correct / test_total

        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # -----------------------------
        # EARLY STOPPING
        # -----------------------------

        if decision["stop"]:

            print("Training stopped by NeuroOptAI controller.")

            break

    print("Training completed.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    train()
