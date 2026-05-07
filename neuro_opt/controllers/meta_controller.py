class MetaController:
    """
    High-level adaptive optimization controller.

    The MetaController coordinates:
    - stopping decisions
    - learning-rate adaptation
    - optimizer switching
    - future meta-strategies

    Core principle:
        continue only if expected benefit
        exceeds optimization cost and risk.
    """

    def __init__(
        self,
        stop_controller,
        lr_controller=None,
        optimizer_switcher=None,
    ):
        self.stop_controller = stop_controller
        self.lr_controller = lr_controller
        self.optimizer_switcher = optimizer_switcher

        self.history = []

    def decide(
        self,
        loss,
        optimizer=None,
        epoch=None,
        compute_cost=1.0,
    ):
        """
        Main adaptive decision method.
        """

        self.history.append(loss)

        decision = {
            "continue": True,
            "stop": False,
            "adjust_lr": False,
            "switch_optimizer": False,
            "new_optimizer": None,
        }

        # ---------------------------------------------------
        # STOPPING DECISION
        # ---------------------------------------------------

        if self.stop_controller is not None:
            should_stop = self.stop_controller.should_stop(
                loss=loss,
                compute_cost=compute_cost,
            )

            if should_stop:
                decision["continue"] = False
                decision["stop"] = True

                return decision

        # ---------------------------------------------------
        # LEARNING RATE ADAPTATION
        # ---------------------------------------------------

        if (
            self.lr_controller is not None
            and optimizer is not None
        ):
            self.lr_controller.update(
                optimizer=optimizer,
                loss=loss,
            )

            decision["adjust_lr"] = True

        # ---------------------------------------------------
        # OPTIMIZER SWITCHING
        # ---------------------------------------------------

        if (
            self.optimizer_switcher is not None
            and epoch is not None
        ):
            new_optimizer = self.optimizer_switcher.choose_by_epoch(
                epoch
            )

            decision["switch_optimizer"] = True
            decision["new_optimizer"] = new_optimizer

        return decision

    def reset_history(self):
        """
        Reset optimization history.
        """

        self.history = []

    def get_history(self):
        """
        Return stored loss history.
        """

        return self.history
