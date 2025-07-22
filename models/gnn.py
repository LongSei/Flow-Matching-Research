# Code base: https://github.com/lazaratan/meta-flow-matching/

import torch
import math
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from dataclasses import dataclass
from models.mlp import SkipMLP

@dataclass(eq=False)
class GlobalGNN(nn.Module):
    """
    A Global Graph Neural Network model for flow matching.
    
    Args: 
        D (int): The dimensionality of the output space.
        num_hidden_gnn (int): The number of hidden units in the GNN layers.
        num_layers_gnn (int): The number of GNN layers.
        num_hidden_decoder (int): The number of hidden units in the decoder.
        num_layers_decoder (int): The number of decoder layers.
        t_embedding_dim (int): The dimensionality of the timestep embedding.
        knn_k (int): The number of nearest neighbors to consider for graph construction.
        num_spatial_samples (int): The number of spatial samples to use.
        spatial_feat_scale (float): Scaling factor for spatial features.
    """
    D: int = 2
    num_hidden_gnn: int = 64
    num_layers_gnn: int = 3
    num_hidden_decoder: int = 128
    num_layers_decoder: int = 4
    t_embedding_dim: int = 64
    knn_k: int = 50
    num_spatial_samples: int = 64
    spatial_feat_scale: float = 1.0

    def __post_init__(self):
        super().__init__()
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(self.D, self.num_hidden_gnn))
        for _ in range(self.num_layers_gnn - 1):
            self.gcn_convs.append(GCNConv(self.num_hidden_gnn, self.num_hidden_gnn))
            
        decoder_input_size = (
            self.num_hidden_gnn + self.t_embedding_dim + 2 * self.num_spatial_samples + self.D
        )
        self.decoder = SkipMLP(
            decoder_input_size, self.D, self.num_hidden_decoder, self.num_layers_decoder
        )
        B = torch.randn((self.D, self.num_spatial_samples)) * self.spatial_feat_scale
        self.register_buffer("B", B)

    def embed_source(self, source_samples: torch.Tensor) -> torch.Tensor:
        is_batched = source_samples.dim() > 2
        device = source_samples.device
        
        if is_batched:
             source_samples = source_samples.squeeze(0)

        edge_index = torch_geometric.nn.pool.knn_graph(source_samples.cpu(), k=self.knn_k).to(device)
        z = source_samples

        for i, conv in enumerate(self.gcn_convs):
            z = conv(z, edge_index)
            if i < len(self.gcn_convs) - 1:
                z = F.relu(z)

        z = z.mean(dim=0, keepdim=True)
        return F.normalize(z, p=2, dim=-1)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)  
        
        half_dim = self.t_embedding_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)).to(t.device)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.t_embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def flow(self, embedding: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the flow model by combining embedding, time step, and input features
        with Fourier feature transformation and passing them through a decoder.

        Args:
            embedding (torch.Tensor): A tensor of shape (1, E) or (B, E) representing the global embedding 
                                    or context vector, where E is the embedding dimension.
            t (torch.Tensor): A tensor of shape (1,) or (B, 1) representing the time steps.
            y (torch.Tensor): A tensor of shape (B, D + k) representing the input features, where the first D 
                            dimensions are used for Fourier projection.

        Returns:
            torch.Tensor: A tensor of shape (B, output_dim), which is the output of the decoder applied 
                        to the concatenated input.
        """
        t_emb = self.get_timestep_embedding(t.squeeze(-1) if t.dim() > 1 else t).expand(y.shape[0], -1)
        y_proj = 2.0 * torch.pi * y[:, :self.D] @ self.B
        y_fourier = torch.cat([y_proj.cos(), y_proj.sin()], dim=-1)
        embedding = embedding.expand(y.shape[0], -1)
        z = torch.cat([embedding, t_emb, y_fourier, y], dim=-1)
        return self.decoder(z)

    def update_embedding_for_inference(self, source_samples: torch.Tensor):
        self.embedding = self.embed_source(source_samples).detach()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.flow(self.embedding, t, x)