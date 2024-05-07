import torch
import math
from pathlib import Path
from typing import Union, Dict


class ModelCheckpoint:
    def __init__(self, filepath: str = 'checkpoint.pth', monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = False, save_freq: int = 1):
        """
        Auto-save checkpoints during training.

        Parameters:
            filepath: The location where checkpoints will be saved. If a folder is specified, multiple checkpoints will be saved.
            monitor: The metric to monitor. It takes effect only if passed a `dict`.
            mode: The mode for monitoring ('min' or 'max').
            save_best_only: Whether to save only the best checkpoints based on the monitored metric.
            save_freq: Frequency of saving (only valid if `save_best_only=False`).
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.times = 1
        self.best_value = -math.inf if mode == 'max' else math.inf

    def save(self, **kwargs):
      path = self.filepath
      if path.is_dir():
        path = path / f'checkpoint-{self.times}.pth'
      path.parent.mkdir(parents=True, exist_ok=True)
      model = kwargs.get("model")

      if isinstance(model, nn.DataParallel):
        kwargs["model"] = model.module.state_dict()  # Extract the underlying model
      torch.save(kwargs, str(path))


    def state_dict(self):
        """Return the state for later recovery."""
        return {
            'filepath': str(self.filepath),
            'monitor': self.monitor,
            'save_best_only': self.save_best_only,
            'mode': self.mode,
            'save_freq': self.save_freq,
            'times': self.times,
            'best_value': self.best_value
        }

    def load_state_dict(self, state_dict: dict):
        """Load the state from a saved checkpoint."""
        self.filepath = Path(state_dict['filepath'])
        self.monitor = state_dict['monitor']
        self.save_best_only = state_dict['save_best_only']
        self.mode = state_dict['mode']
        self.save_freq = state_dict['save_freq']
        self.times = state_dict['times']
        self.best_value = state_dict['best_value']

    @staticmethod
    def load_state_dict_ex(model, state_dict):
      """Load state dict into model while handling 'module.' prefix issues."""
      # check for 'module.' prefix and adjust state_dict keys
      new_state_dict = {}
      for key, value in state_dict.items():
        if key.startswith('module.'):
          new_key = key[len('module.'):]
        else:
          new_key = key
        new_state_dict[new_key] = value

      # load adjusted state_dict into the model
      model.load_state_dict(new_state_dict)


    def reset(self):
        """Reset the times counter to its initial state."""
        self.times = 1

    def step(self, metrics: Union[Dict, int, float], **kwargs):
        """
        Determine whether to save a checkpoint based on the given metrics.

        Parameters:
            metrics: A dictionary containing the `monitor` key or a scalar metric.
            kwargs: The data to be saved in the checkpoint.
        """
        if isinstance(metrics, dict):
            metrics = metrics.get(self.monitor, None)
            if metrics is None:
                raise ValueError(f"Monitor key '{self.monitor}' not found in metrics.")

        save_flag = False
        if self.save_best_only:
            if (self.mode == 'min' and metrics <= self.best_value) or (
                    self.mode == 'max' and metrics >= self.best_value):
                self.best_value = metrics
                self.save(**kwargs)
                save_flag = True
        else:
            if self.times % self.save_freq == 0:
                self.save(**kwargs)
                save_flag = True

        self.times += 1
        return save_flag