import copy
import torch


class BranchPruningOptimizer:
    """
    Experimental branch-pruning optimization system.

    The optimizer explores multiple optimization branches,
    evaluates their short-term performance,
    and prunes low-potential branches dynamically.
    """

    def __init__(
        self,
        model,
        optimizer_factory,
        loss_fn,
        device="cpu"
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.loss_fn = loss_fn
        self.device = device

    def evaluate_branches(
        self,
        train_step_fn,
        batch,
        branch_configs,
        prune_ratio=0.5
    ):
        """
        Evaluate multiple optimization branches and prune weak branches.

        Parameters
        ----------
        train_step_fn : function
            Function that performs one training step.

        batch : tuple
            Training batch (x, y).

        branch_configs : list
            List of optimizer configuration dictionaries.

        prune_ratio : float
            Fraction of branches to prune.

        Returns
        -------
        best_branches : list
            Remaining top-performing branches.
        """

        results = []

        for config in branch_configs:

            # Clone model
            model_copy = copy.deepcopy(self.model).to(self.device)

            # Create optimizer
            optimizer = self.optimizer_factory(
                model_copy,
                config
            )

            # Evaluate branch
            loss = train_step_fn(
                model_copy,
                optimizer,
                self.loss_fn,
                batch,
                self.device
            )

            results.append({
                "config": config,
                "loss": loss,
                "model_state": copy.deepcopy(
                    model_copy.state_dict()
                )
            })

        # Sort by loss
        results.sort(key=lambda x: x["loss"])

        # Number of branches to keep
        keep_n = max(
            1,
            int(len(results) * (1 - prune_ratio))
        )

        best_branches = results[:keep_n]

        # Load best branch into main model
        self.model.load_state_dict(
            best_branches[0]["model_state"]
        )

        return best_branches


def default_train_step(
    model,
    optimizer,
    loss_fn,
    batch,
    device="cpu"
):
    """
    Default training step.
    """

    model.train()

    x, y = batch

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    predictions = model(x)

    loss = loss_fn(predictions, y)

    loss.backward()

    optimizer.step()

    return loss.item()


if __name__ == "__main__":

    print("BranchPruningOptimizer module loaded.")
