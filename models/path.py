import torch
import torch.nn as nn
from typing import Optional, Union
from torch import Tensor
from torch.nn import functional as F
import math

class OptimalTransport(): 
    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.epsilon = 1e-5
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
class OptimalTransportWithRandomResponse:
    """
    A class to implement Conditional Flow Matching with a differentially private
    loss function using Sliced Score Matching and Randomized Response.
    
    The core idea is to project the model's output vector field and the target
    vector field onto several random directions. Then, for each sample, we
    randomly select one of these projections according to epsilon-DP randomized
    response probabilities. The loss is the mean squared error between the
    model's selected projection and the target's selected projection for that
    same randomly chosen direction. This ensures the training signal is noisy
    but unbiased.
    """
    def __init__(self, 
                 k: int = 10,
                 sigma_min: float = 0.001, 
                 epsilon: float = 1.0):
        """
        Initializes the OT-RR framework.

        Args:
            k (int, optional): The total number of random directions to project onto. 
                               Defaults to 10.
            sigma_min (float, optional): The minimum noise level for the OT path. 
                                         Defaults to 0.001.
            epsilon (float, optional): The privacy budget ε for differential privacy.
                                       Defaults to 1.0.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.epsilon = epsilon
        self.k = k
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Probability of choosing the "true" direction (the first one)
        self.p_true = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
        # Probability of choosing any of the other k-1 "fake" directions
        self.p_fake = 1.0 / (math.exp(epsilon) + k - 1)
        
        print(f"Initialized OT-RR on device: {self.device}")
        print(f"Privacy budget ε={self.epsilon}, Directions k={self.k}")
        print(f"Probability of true projection: {self.p_true:.4f}")
        print(f"Probability of any fake projection: {self.p_fake:.4f}")

    def _sample_random_directions(self, 
                                  batch_size: int, 
                                  dim: int) -> torch.Tensor:
        """
        Samples k random unit vectors for each item in the batch.

        Args:
            batch_size (int): The number of samples in the current batch.
            dim (int): The dimension of the vector space (e.g., flattened image size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, k, dim) containing
                          the random directions.
        """
        # Sample from a standard normal distribution
        dirs = torch.randn(batch_size, self.k, dim, device=self.device)
        # Normalize to get unit vectors
        dirs = F.normalize(dirs, p=2, dim=-1)
        return dirs
    
    def _get_ot_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the Optimal Transport (OT) path between noise (x0) and data (x1).
        This defines the interpolated sample x_t at time t.

        Args:
            x0 (torch.Tensor): The starting point of the path (e.g., Gaussian noise).
            x1 (torch.Tensor): The ending point of the path (e.g., real data).
            t (torch.Tensor): A tensor of time steps in [0, 1] for each sample.

        Returns:
            torch.Tensor: The interpolated state x_t.
        """
        # Ensure t has the correct shape for broadcasting (e.g., [B, 1, 1, 1])
        while len(t.shape) < len(x0.shape):
            t = t.unsqueeze(-1)
            
        # Linear interpolation for the OT path
        return (1 - (1 - self.sigma_min) * t) * x0 + t * x1
    
    def loss(self,
             vector_field_model: nn.Module,
             x1: torch.Tensor,
             class_condition: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the differentially private loss using Sliced Score Matching
        with Randomized Response.

        Args:
            vector_field_model (nn.Module): The neural network that predicts the vector field.
            x1 (torch.Tensor): A batch of real data samples (the target).
            class_condition (torch.Tensor, optional): Class conditions for the model. 
                                                      Defaults to None.

        Returns:
            torch.Tensor: A scalar tensor representing the final loss.
        """
        batch_size = x1.shape[0]
        
        # 1. Sample time t uniformly from [0, 1]
        # Use a small epsilon to avoid t=1, where the vector field might be undefined.
        time_epsilon = 1e-5
        t = torch.rand(batch_size, device=x1.device) * (1 - time_epsilon)
        
        # 2. Sample initial noise x0 from a standard normal distribution
        x0 = torch.randn_like(x1)

        # 3. Compute the interpolated state x_t along the OT path
        x_t = self._get_ot_path(x0, x1, t)

        # 4. Get the model's predicted vector field v_θ(t, x_t)
        if class_condition is None:
            v_predicted = vector_field_model(t, x_t)
        else: 
            v_predicted = vector_field_model(t, x_t, class_condition)

        # 5. Compute the true target vector field u_t(x_t) = x1 - (1-σ_min)x0
        v_target = x1 - (1 - self.sigma_min) * x0
        
        # 6. Flatten vectors for projection
        v_predicted_flat = v_predicted.view(batch_size, -1)
        v_target_flat = v_target.view(batch_size, -1)
        dim = v_predicted_flat.shape[1]
        
        # 7. Sample k random directions for the projections
        rand_dirs = self._sample_random_directions(batch_size, dim)  # Shape: (B, k, D)

        # 8. Project both predicted and target vectors onto all k directions
        # torch.sum(v.unsqueeze(1) * dirs, dim=-1) is equivalent to a batched dot product
        proj_predicted = torch.sum(v_predicted_flat.unsqueeze(1) * rand_dirs, dim=-1) # Shape: (B, k)
        proj_target = torch.sum(v_target_flat.unsqueeze(1) * rand_dirs, dim=-1)       # Shape: (B, k)

        # --- Corrected Randomized Response Logic ---
        # 9. For each sample, make ONE random choice of direction based on RR probabilities.
        
        # Create a mask to decide whether to keep the "true" projection (at index 0)
        keep_true_mask = torch.rand(batch_size, device=self.device) < self.p_true

        # For samples where we don't keep the true one, randomly choose a "fake" index from {1, ..., k-1}
        fake_indices = torch.randint(1, self.k, (batch_size,), device=self.device)

        # The final chosen index is 0 if keep_true_mask is true, otherwise it's the fake_index
        chosen_indices = torch.where(keep_true_mask, 
                                     torch.zeros_like(fake_indices), 
                                     fake_indices) # Shape: (B,)

        # 10. Use the SAME chosen indices to select the projections for BOTH the model and the target.
        # This ensures we are comparing the model's output to the correct ground truth.
        
        # Gather the projections from the model's output using the chosen indices
        rr_proj_predicted = proj_predicted.gather(1, chosen_indices.unsqueeze(-1)).squeeze(-1)
        
        # Gather the projections from the target using the exact same indices
        rr_proj_target = proj_target.gather(1, chosen_indices.unsqueeze(-1)).squeeze(-1)
        
        # 11. Compute the final loss. Detach the target to prevent gradients flowing through it.
        loss = torch.mean((rr_proj_predicted - rr_proj_target.detach()) ** 2)

        return loss