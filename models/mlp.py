# Code base: https://github.com/lazaratan/meta-flow-matching/

import torch 
import torch.nn as nn

class SkipMLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) with skip connections."""
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            hidden_size: int,
            num_layers: int, 
            act_fn: nn.Module = nn.SELU()
        ):
        """
        Initializes the SkipMLP model. Used for input (BATCH_SIZE, INPUT_SIZE) and output (BATCH_SIZE, OUTPUT_SIZE).
        
        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output features.
            hidden_size (int): The size of the hidden layers.
            num_layers (int): The number of hidden layers.
            act_fn (nn.Module, optional): The activation function to use. Defaults to nn.SELU().
        """
        super().__init__()
        self.activation = act_fn
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
    
    def flatten_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flattens the input tensor to ensure it has the correct shape for the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (BATCH_SIZE, INPUT_SIZE).
        
        Returns:
            torch.Tensor: Flattened tensor of shape (BATCH_SIZE, INPUT_SIZE).
        """
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_data(x)
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out)) + out
        return self.output_layer(out)