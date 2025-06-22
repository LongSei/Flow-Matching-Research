import torch
import torch.nn as nn
from typing import Optional, Union
from torch import Tensor
from torch.nn import functional as F

class OptimalTransport(): 
    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.epsilon = 1e-5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Maching with Optimal Transport

        Args:
            x (torch.Tensor): new state of sample
            x_1 (torch.Tensor): condition state to generate new sample
            t (torch.Tensor): random at step t for Mean and Variance of OT Formulation with range (0, 1)

        Returns:
            torch.Tensor: Flow matching with optimal transport at x with condition x_1 and step t
        """

        t = t[:, :, None, None]
        return (1 - (1 - self.sigma_min) * t) * x + t * x_1

    def loss(self,
             vector_field_t: nn.Module,
             x_1: torch.Tensor,
             class_condition: torch.Tensor=None) -> torch.Tensor:
        """
        Compute the loss of the flow matching with optimal transport

        Args:
            vector_field_t (nn.Module): Network to calculate Vector Field at step t
            x_1 (torch.Tensor): Condition state x_1
            class_condition (int): Class condition for a batch of samples

        Returns:
            torch.Tensor: Loss of the flow matching with optimal transport
        """

        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.epsilon)
        t = t[:, None]

        # Random Noise
        x_0 = torch.randn_like(x_1)

        # Calculate the vector field at step t
        psi_t = self.psi_t(x_0, x_1, t)

        # Calculate the vector field at step t with psi_t as input
        if class_condition is None:
            v_psi = vector_field_t(t[:, 0], psi_t)
        else: 
            v_psi = vector_field_t(t[:, 0], psi_t, class_condition)

        # Calculate derivative of psi_t at x_0 respect to t
        d_psi = x_1 - (1 - self.sigma_min) * x_0

        # Calculate the loss
        return torch.mean((v_psi - d_psi) ** 2)