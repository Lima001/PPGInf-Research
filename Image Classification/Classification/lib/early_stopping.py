import numpy as np

class EarlyStopping:
    """
    Monitors a metric and stops training when it stops improving.
    """
    def __init__(self, patience, mode, min_delta):
        """
        Args:
            patience: How many epochs to wait after last time the monitored metric improved.
            mode: The direction of improvement. 'max' for metrics like accuracy, 'min' for metrics like loss.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf if mode == 'max' else np.inf
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        """Returns True if training should be stopped."""
        return self._should_stop

    def step(self, metric_value: float) -> bool:
        """
        Updates the internal state with the latest metric value.

        Args:
            metric_value: The validation metric from the current epoch.

        Returns:
            bool: True if the metric improved, False otherwise.
        """
        improved = False
        if self.mode == 'max':
            if metric_value > self.best_score + self.min_delta:
                self.best_score = metric_value
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else: # mode == 'min'
            if metric_value < self.best_score - self.min_delta:
                self.best_score = metric_value
                self.counter = 0
                improved = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self._should_stop = True
        
        return improved