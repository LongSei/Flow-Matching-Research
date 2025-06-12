import torch 
import torchcfm 
import torch.nn as nn
import math
import torch.nn.functional as F

class SkipMLP_Layer(nn.Module): 
    def __init__(self, 
                 input_size: int=None, 
                 hidden_size: int=None,
                 output_size: int=None,
                 num_layers: int=None, 
                 activation_function: nn.Module=nn.SELU):
        super(SkipMLP_Layer, self).__init__()
        
        self.num_layers = num_layers
        self.activation_function = activation_function()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layer.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x): 
        input_out = self.input_layer(x)
        
        hidden_out = input_out
        for layer_index in range(self.num_layers - 1):
            layer = self.hidden_layer[layer_index]
            hidden_out = self.activation_function(layer(hidden_out) + hidden_out)
        
        output_out = self.output_layer(hidden_out)
        return output_out

class NormalMLP_Layer(nn.Module):
    def __init__(self, 
                 input_size: int=None, 
                 hidden_size: int=None,
                 output_size: int=None,
                 num_layers: int=None, 
                 activation_function: nn.Module=nn.SELU):
        super(NormalMLP_Layer, self).__init__()
        
        self.num_layers = num_layers
        self.activation_function = activation_function()
        
        layers = []
        prev_size = input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation_function)
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.net(x)

class ConvolutionBlock_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, t_embedding_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        self.time_proj = nn.Linear(t_embedding_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv(x)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # shape: (B, C, 1, 1)
        return h + t

class TimeEmbedding_NeuralNetwork(nn.Module):
    def __init__(self, 
                 input_size: int=28*28, 
                 hidden_size: int=512,
                 output_size: int=10,
                 t_embedding_dim: int=64,
                 num_hidden_layers: int=3) -> None: 
        super(TimeEmbedding_NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.t_embedding_dim = t_embedding_dim
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        prev_size = input_size + t_embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        
    # TODO - add support for different embedding types 
    def get_time_step_embedding(self, t: torch.Tensor, embedding_type: str = 'sinusoidal') -> torch.Tensor:
        """
        Generate time step embeddings.

        Args:
            t: Tensor of shape (B,), time steps (floats or ints)
            embedding_type: Type of embedding ('sinusoidal' or 'fourier')

        Returns:
            Tensor of shape (B, t_embedding_dim)
        """
        if embedding_type != 'sinusoidal':
            raise ValueError("Unsupported embedding type. Use 'sinusoidal'.")

        half_dim = self.t_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]  # shape: (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.t_embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb  # shape: (B, t_embedding_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, input_size).
            t (torch.Tensor): Time step tensor of shape (B,).

        Returns:
            torch.Tensor: Output tensor of shape (B, output_size).
        """
        t_emb = self.get_time_step_embedding(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)