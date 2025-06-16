from optimizers import Optimizer_Base

class LearningRateScheduler:
    def __init__(self, optimizer: Optimizer_Base, decay: float = 2, window: int = 5):
        self.optimizer = optimizer
        self.decay: float = decay
        self.window: int = window
        self.losses: list[float] = []

    def at_epoch_end(self, loss: float):
        if len(self.losses) >= self.window:
            self.losses.pop()
        
        self.losses.append(loss)
        if self._plateau_dect():
            self.optimizer.learning_rate /= self.decay
    
    def _plateau_dect(self) -> bool:
        avg = sum(self.losses) / len(self.losses)
        std = sum((x - avg)**2 for x in self.losses) / (len(self.losses) - 1)
        if std < 0.1: 
            return True
        return False