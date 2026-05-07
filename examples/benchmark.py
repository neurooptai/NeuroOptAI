import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neuro_opt.controllers.cost_aware_early_stopping import (
    CostAwareEarlyStopping,
)

from neuro_opt.controllers.adaptive_lr_controller import (
    AdaptiveLRController,
)

from neuro_opt.controllers.meta_controller import (
    MetaController,
)


class TinyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predicted = pred.argmax(dim=1)

        correct += (predicted == y).sum().item()

        total += y.size(0)

    epoch_time = time.time() - start_time

    avg_loss = total_loss / len(loader)

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time


def benchmark():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    model = TinyNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    stopper = CostAwareEarlyStopping(
        patience=5,
        min_gain=1e-3,
        cost_weight=1e-5
    )

    lr_controller = AdaptiveLRController()

    controller = MetaController(
        stopper,
        lr_controller
    )

    results = []

    max_epochs = 20

    for epoch in range(max_epochs):

        loss, accuracy, epoch_time = train_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        decision = controller.decide(
            loss=loss,
            optimizer=optimizer,
            compute_cost=len(loader)
        )

        result = {
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy,
            "time": epoch_time,
            "lr": current_lr,
            "decision": decision
        }

        results.append(result)

        print("-" * 60)

        print(f"Epoch:      {epoch + 1}")
        print(f"Loss:       {loss:.4f}")
        print(f"Accuracy:   {accuracy:.2f}%")
        print(f"Time:       {epoch_time:.2f}s")
        print(f"LR:         {current_lr:.6f}")
        print(f"Decision:   {decision}")

        if decision["stop"]:

            print("\nTraining stopped by cost-aware controller.")

            break

    print("\n================ FINAL RESULTS ================\n")

    best = max(results, key=lambda x: x["accuracy"])

    print(f"Best Accuracy: {best['accuracy']:.2f}%")
    print(f"Best Loss:     {best['loss']:.4f}")
    print(f"Epoch:         {best['epoch']}")

    total_time = sum(r["time"] for r in results)

    print(f"Total Time:    {total_time:.2f}s")

    print("\nBenchmark completed.")


if __name__ == "__main__":

    benchmark()
