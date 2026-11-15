"Losses for FNO and baseline"
import torch
import torch.nn.functional as F

def field_mse(pred: torch.Tensor, target: torch.Tensor, skip_sensors: int=1) -> torch.Tensor:
    return F.mse_loss(pred[..., skip_sensors:, :], target[..., skip_sensors:, :])