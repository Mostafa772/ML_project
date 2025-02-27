import numpy as np

class Regularization:
    def __call__(self, weights: np.ndarray | None) -> np.ndarray:
        raise NotImplemented

class Lasso(Regularization):
    def __init__(self, l: float) -> None:
        super().__init__()
        self.l = l #l is lambda, the regularization strenght

    def __call__(self, weights: np.ndarray | None) -> np.ndarray:
        if weights is None:
            return np.array(0)
        
        reg = self.l * np.sum(np.linalg.norm(weights, ord=1))
        if isinstance(reg, np.ndarray):
            return reg

        return np.array(reg) 

class Tikhonov(Regularization):
    def __init__(self, l: float) -> None:
        super().__init__()
        self.l = l #lambda, the regularization strenght

    def __call__(self, weights: np.ndarray | None) -> np.ndarray:
        if weights is None:
            return np.array(0)

        return 2 * self.l * np.sum(np.square(weights))