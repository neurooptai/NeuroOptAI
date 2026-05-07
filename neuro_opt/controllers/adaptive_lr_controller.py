class AdaptiveLRController:
    """
    Adaptive Learning Rate Controller

    Dynamically adjusts the learning rate according to
    optimization progress and recent loss behavior.
    """

    def __init__(
        self,
        factor_down=0.5,
        factor_up=1.05,
        patience=3,
        min_lr=1e-6,
        max_lr=1.0
    ):
        self.factor_down = factor_down
        self.factor_up = factor_up
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.loss_history = []

    def update(self, optimizer, current_loss):
        """
        Update optimizer learning rate based on loss evolution.
        """

        self.loss_history.append(current_loss)

        if len(self.loss_history) < self.patience + 1:
            return

        recent_losses = self.loss_history[-self.patience:]
        previous_loss = self.loss_history[-self.patience - 1]

        # If optimization stagnates -> reduce LR
        if min(recent_losses) >= previous_loss:
            self._scale_lr(optimizer, self.factor_down)

        # If optimization improves -> slightly increase LR
        else:
            self._scale_lr(optimizer, self.factor_up)

    def _scale_lr(self, optimizer, factor):
        """
        Scale learning rate safely within bounds.
        """

        for param_group in optimizer.param_groups:

            old_lr = param_group["lr"]

            new_lr = old_lr * factor

            # Clamp learning rate
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))

            param_group["lr"] = new_lr

            print(
                f"[AdaptiveLRController] "
                f"LR updated: {old_lr:.6f} -> {new_lr:.6f}"
            )

    def get_history(self):
        """
        Return stored loss history.
        """

        return self.loss_history
