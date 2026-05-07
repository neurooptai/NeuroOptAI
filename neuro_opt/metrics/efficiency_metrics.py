import time
import torch


class EfficiencyMetrics:
    """
    Utility class for tracking optimization efficiency metrics
    during neural network training.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

        self.loss_history = []
        self.accuracy_history = []

        self.total_steps = 0
        self.total_samples = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def add_loss(self, loss):
        self.loss_history.append(loss)

    def add_accuracy(self, accuracy):
        self.accuracy_history.append(accuracy)

    def add_step(self, batch_size=1):
        self.total_steps += 1
        self.total_samples += batch_size

    def training_time(self):
        if self.start_time is None or self.end_time is None:
            return None

        return self.end_time - self.start_time

    def average_loss(self):
        if not self.loss_history:
            return None

        return sum(self.loss_history) / len(self.loss_history)

    def best_loss(self):
        if not self.loss_history:
            return None

        return min(self.loss_history)

    def average_accuracy(self):
        if not self.accuracy_history:
            return None

        return sum(self.accuracy_history) / len(self.accuracy_history)

    def best_accuracy(self):
        if not self.accuracy_history:
            return None

        return max(self.accuracy_history)

    def samples_per_second(self):
        elapsed = self.training_time()

        if elapsed is None or elapsed == 0:
            return None

        return self.total_samples / elapsed

    def optimization_efficiency_score(self):
        """
        Experimental efficiency score.

        Higher accuracy and lower loss improve the score.
        Longer training time penalizes the score.
        """

        avg_acc = self.average_accuracy() or 0
        avg_loss = self.average_loss() or 1
        elapsed = self.training_time() or 1

        score = (avg_acc / avg_loss) / elapsed

        return score

    def gpu_memory_usage_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)

        return 0

    def summary(self):
        return {
            "training_time_seconds": self.training_time(),
            "average_loss": self.average_loss(),
            "best_loss": self.best_loss(),
            "average_accuracy": self.average_accuracy(),
            "best_accuracy": self.best_accuracy(),
            "samples_per_second": self.samples_per_second(),
            "efficiency_score": self.optimization_efficiency_score(),
            "gpu_memory_mb": self.gpu_memory_usage_mb(),
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
        }
