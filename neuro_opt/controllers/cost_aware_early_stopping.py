class CostAwareEarlyStopping:
    """
    Cost-aware early stopping controller.

    Stops training when the optimization gain
    is smaller than the estimated computational cost.
    """

    def __init__(
        self,
        patience=5,
        min_gain=1e-3,
        cost_weight=1e-5
    ):
        self.patience = patience
        self.min_gain = min_gain
        self.cost_weight = cost_weight

        self.loss_history = []

    def should_stop(
        self,
        current_loss,
        compute_cost=1.0
    ):
        """
        Decide whether training should stop.

        Parameters
        ----------
        current_loss : float
            Current training loss.

        compute_cost : float
            Estimated computational cost.

        Returns
        -------
        bool
            True if training should stop.
        """

        self.loss_history.append(current_loss)

        # Not enough history yet
        if len(self.loss_history) <= self.patience:
            return False

        previous_loss = self.loss_history[
            -self.patience - 1
        ]

        gain = previous_loss - current_loss

        required_gain = (
            self.min_gain +
            self.cost_weight * compute_cost
        )

        # Stop if gain is too small
        return gain < required_gain

    def reset(self):
        """
        Reset internal state.
        """

        self.loss_history = []

    def get_statistics(self):
        """
        Return controller statistics.
        """

        if len(self.loss_history) == 0:
            return {}

        return {
            "latest_loss": self.loss_history[-1],
            "best_loss": min(self.loss_history),
            "history_length": len(self.loss_history)
        }
