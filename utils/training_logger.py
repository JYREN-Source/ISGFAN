# utils/training_logger.py

import os
import json
from collections import defaultdict
import numpy as np

class TrainingLogger:

    def __init__(self, stage_name):

        self.stage_name = stage_name
        self.losses = defaultdict(list)
        self.accuracies = defaultdict(list)

    def log_losses(self, loss_dict):
        for loss_name, loss_value in loss_dict.items():
            self.losses[loss_name].append(float(loss_value))

    def log_accuracies(self, acc_dict):
        for acc_name, acc_value in acc_dict.items():
            self.accuracies[acc_name].append(float(acc_value))

    def save_to_file(self):
        save_dir = f"training_logs/stage_{self.stage_name}"
        os.makedirs(save_dir, exist_ok=True)

        for loss_name, loss_values in self.losses.items():
            file_path = os.path.join(save_dir, f"{loss_name}.txt")
            np.savetxt(file_path, np.array(loss_values))

        for acc_name, acc_values in self.accuracies.items():
            file_path = os.path.join(save_dir, f"{acc_name}.txt")
            np.savetxt(file_path, np.array(acc_values))

        config = {
            'loss_types': list(self.losses.keys()),
            'accuracy_types': list(self.accuracies.keys()),
            'iterations': len(next(iter(self.losses.values()))) if self.losses else 0
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def get_latest_metrics(self):
        latest_losses = {name: values[-1] if values else None
                         for name, values in self.losses.items()}
        latest_accs = {name: values[-1] if values else None
                       for name, values in self.accuracies.items()}
        return latest_losses, latest_accs