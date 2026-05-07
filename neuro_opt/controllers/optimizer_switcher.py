import torch


class OptimizerSwitcher:
    """
    Dynamic optimizer selection system.

    This controller allows switching optimizers
    depending on the current training phase.
    """

    def __init__(self, model, initial_lr=1e-3):
        self.model = model
        self.initial_lr = initial_lr

    def make(self, optimizer_name):
        """
        Create optimizer by name.
        """

        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.initial_lr
            )

        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.initial_lr
            )

        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.initial_lr,
                momentum=0.9
            )

        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.initial_lr
            )

        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}"
            )

    def choose_by_epoch(self, epoch):
        """
        Simple adaptive optimizer strategy.

        Early epochs:
            AdamW for fast convergence

        Middle epochs:
            Adam for stability

        Late epochs:
            SGD for fine tuning
        """

        if epoch < 5:
            return self.make("adamw")

        elif epoch < 15:
            return self.make("adam")

        else:
            return self.make("sgd")
