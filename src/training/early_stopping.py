"""Validation macro F1 early stopping and best checkpoint tracker."""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping tracker that halts training when validation metric fails to improve.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_metric: float) -> Tuple[bool, bool]:
        """
        Evaluate current metric against historical best.

        Returns:
            Tuple of (should_stop: bool, is_best: bool).
        """
        if self.best_score is None:
            self.best_score = current_metric
            return False, True

        if self.mode == "max":
            improved = (current_metric - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_metric) > self.min_delta

        if improved:
            self.best_score = current_metric
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True, False
            return False, False