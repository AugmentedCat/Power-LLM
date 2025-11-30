from typing import Dict, Optional
from collections import deque


class Recorder(object):
    def __init__(self, rolling_window: int = 100):
        self.metrics = {}
        self.batch_metrics = {}
        self.rolling_window = rolling_window
        self.rolling_metrics = {}  # Store rolling averages over individual batch updates
        self.all_batch_values = {}  # Store all individual batch values for rolling avg
        self.latest_values = {}  # Store latest value for non-smoothed metrics

    def record(self, metrics: Dict[str, float], scope: Optional[str] = None):
        for name, value in metrics.items():
            name = f'{scope}/{name}' if scope else name

            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

            # Store latest value
            self.latest_values[name] = value

            # Only add train metrics to rolling window for smoothing
            if scope == 'train':
                if name not in self.all_batch_values:
                    self.all_batch_values[name] = deque(maxlen=self.rolling_window)
                self.all_batch_values[name].append(value)

    def stamp(self, step: int = 0):
        for name, values in self.batch_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []

            # Add the average of metrics values in the batch.
            avg_value = sum(values) / len(values)
            self.metrics[name].append((step, avg_value))

        self.batch_metrics.clear()

    def format(self, fstring: str) -> str:
        # Use rolling average for train metrics, raw value for eval metrics
        display_values = {}
        for k, v in self.latest_values.items():
            key = k.replace('/', '_')
            # Use rolling average for train metrics, latest value for eval
            if k.startswith('train/') and k in self.all_batch_values and len(self.all_batch_values[k]) > 0:
                display_values[key] = sum(self.all_batch_values[k]) / len(self.all_batch_values[k])
            else:
                display_values[key] = v

        return fstring.format(**display_values)
