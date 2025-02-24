import torch
from torch.optim.lr_scheduler import _LRScheduler

class SVALRScheduler(_LRScheduler):
    def __init__(self, optimizer, gamma=0.6, lr_floor=1000, last_epoch=-1):
        self.gamma = gamma
        self.num_updates = 0  # Tracks number of optimizer updates
        self.lr_floor = lr_floor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current step."""
        # if self.num_updates == 0:  # Avoid division by zero
        #     return [base_lr for base_lr in self.base_lrs]
        d = min((self.num_updates) ** self.gamma, self.lr_floor)
        return [(1 / d) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update the num_updates counter and adjust the learning rate."""
        self.num_updates += 1
        super().step(epoch)