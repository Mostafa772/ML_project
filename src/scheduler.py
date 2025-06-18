from optimizers import Optimizer_Base

class LearningRateScheduler:
    def __init__(self, optimizer: Optimizer_Base, decay: float = 1.4, window: int = 30, threshold: float = 0.006):
        self.optimizer: Optimizer_Base = optimizer
        self.decay: float = decay
        self.window: int = max(window, 1)
        self.threshold: float = threshold
        self.losses: list[float] = []

    def at_epoch_end(self, loss: float):
        if len(self.losses) >= self.window:
            self.losses.pop(0)
        
        self.losses.append(loss)
        if len(self.losses) == self.window and self._plateau_detect():
            self.optimizer.current_learning_rate = self.optimizer.current_learning_rate / self.decay
            self.optimizer.learning_rate = self.optimizer.learning_rate / self.decay
    
    def _plateau_detect(self) -> bool:
        avg = sum(self.losses) / len(self.losses)
        std = (sum((x - avg) ** 2 for x in self.losses) / (len(self.losses) - 1)) ** 0.5
        return std < self.threshold