import math
from typing import Optional, Callable, List
import torch
from torch.optim import Optimizer

class ValAwareOneCycle():
    """
    One-cycle style per-step baseline trend with epoch-level modulation based on validation loss delta.

    Usage:
      scheduler = ValAwareOneCycle(optimizer, total_steps=..., max_lr=..., div_factor=25,
                                   pct_start=0.3, final_div_factor=1e4, trend='cosine',
                                   sensitivity=5.0, min_mod=0.5, max_mod=1.5)
      for step in train_steps:
          optimizer.step()
          scheduler.step_train()           # call every optimizer.step (per batch)
      # at epoch end:
      scheduler.step_epoch(val_loss)       # call once per epoch to update modulation
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_lr: float,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        pct_start: float = 0.3,
        trend: str = "cosine",            # "cosine" or "linear"
        sensitivity: float = 5.0,         # larger => stronger reaction to val-loss delta
        min_mod: float = 0.5,             # minimum modulation multiplier
        max_mod: float = 1.5,             # maximum modulation multiplier
        eps: float = 1e-8
    ):
        if trend not in ("cosine", "linear"):
            raise ValueError("trend must be 'cosine' or 'linear'")
            
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.max_lr = max_lr
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.pct_start = float(pct_start)
        self.trend = trend
        self.sensitivity = float(sensitivity)
        self.min_mod = float(min_mod)
        self.max_mod = float(max_mod)
        self.eps = eps

        self.start_lr = max_lr / self.div_factor
        self.end_lr = self.start_lr / self.final_div_factor

        # per-parameter-group scale (keep same shape as optimizer.param_groups)
        self.base_lrs = [self.start_lr for _ in optimizer.param_groups]

        self.step_num = 0
        self.epoch = 0
        self.modulation = 1.0
        self.prev_val_loss: Optional[float] = None

        # initialize optimizer lrs to start_lr * modulation
        self._set_lrs([self.start_lr * self.modulation] * len(self.base_lrs))

    def _get_progress(self):
        return min(1.0, max(0.0, self.step_num / max(1, self.total_steps - 1)))

    def _anneal(self, start: float, end: float, progress: float) -> float:
        if self.trend == "linear":
            return start + (end - start) * progress
            
        else:  # cosine
            cos_out = 0.5 * (1 + math.cos(math.pi * progress))
            return end + (start - end) * cos_out

    def _baseline_lr(self, progress: float) -> List[float]:
        # two phases: 0..pct_start increase start->max, pct_start..1 decrease max->end
        if progress <= self.pct_start:
            # progress within phase1 mapped 0..1
            p = progress / max(self.eps, self.pct_start)
            lr = self._anneal(self.start_lr, self.max_lr, p)
            
        else:
            p = (progress - self.pct_start) / max(self.eps, 1.0 - self.pct_start)
            lr = self._anneal(self.max_lr, self.end_lr, p)
            
        return [lr for _ in self.base_lrs]

    def _set_lrs(self, lrs: List[float]):
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def step_train(self):
        """
        Step per optimizer.step() (per batch). Computes baseline by progress and applies modulation.
        """
        self.step_num += 1
        progress = self._get_progress()
        base_lrs = self._baseline_lr(progress)
        mod_lrs = [max(0.0, lr * self.modulation) for lr in base_lrs]
        self._set_lrs(mod_lrs)

    def _default_modulation_fn(self, delta: float) -> float:
        """
        Map val-loss delta -> multiplicative modulation.
        delta = prev_val_loss - curr_val_loss
        Positive delta => validation improved => boost lr up to max_mod.
        Negative delta => validation worsened => reduce lr down to min_mod.
        Uses a sigmoid-like mapping controlled by sensitivity.
        """
        # scale delta (might be small), sensitivity increases effect
        s = self.sensitivity * delta
        # logistic mapping to (0,1)
        sigmoid = 1.0 / (1.0 + math.exp(-s))
        
        # center at 0.5 -> map to [min_mod, max_mod]
        return self.min_mod + (self.max_mod - self.min_mod) * sigmoid

    def step_epoch(self, val_loss: float):
        """
        Call at epoch end with the validation loss for that epoch.
        Updates modulation used for subsequent step_train calls.
        """
        if self.prev_val_loss is None:
            # first epoch: initialize prev and do not change modulation
            self.prev_val_loss = float(val_loss)
            self.epoch += 1
            
            return

        delta = float(self.prev_val_loss) - float(val_loss)
        new_mod = self._default_modulation_fn(delta)
        # optional: smooth updates between old modulation and new_mod (momentum)
        alpha = 0.6  # smoothing factor: higher -> more weight to new_mod
        self.modulation = (1 - alpha) * self.modulation + alpha * new_mod

        self.prev_val_loss = float(val_loss)
        self.epoch += 1

        # optionally update optimizer lr immediately to reflect new modulation at epoch boundary
        progress = self._get_progress()
        base_lrs = self._baseline_lr(progress)
        mod_lrs = [max(0.0, lr * self.modulation) for lr in base_lrs]
        self._set_lrs(mod_lrs)

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "epoch": self.epoch,
            "modulation": self.modulation,
            "prev_val_loss": self.prev_val_loss,
            "optimizer_state": [pg["lr"] for pg in self.optimizer.param_groups],
        }

    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]
        self.epoch = sd["epoch"]
        self.modulation = sd["modulation"]
        self.prev_val_loss = sd["prev_val_loss"]
        # restore optimizer lr if present
        if "optimizer_state" in sd:
            lrs = sd["optimizer_state"]
            if len(lrs) == len(self.optimizer.param_groups):
                self._set_lrs(lrs)
