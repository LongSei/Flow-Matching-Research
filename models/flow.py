import sys 
from pathlib import Path
current_directory = Path(__file__).absolute().parent
parent_directory = current_directory.parent
sys.path.append(str(parent_directory))

import torch
import torch.nn as nn
from models.components import SkipMLP_Layer, NormalMLP_Layer, TimeEmbedding_NeuralNetwork, ConvolutionBlock_Layer
            

class UNet(nn.Module):
    def __init__(self, 
                 input_channels: int= 1,
                 base_channels: int = 64,
                 output_channels: int = 10,
                 t_embedding_dim: int = 128,
                ) -> None: 
        super().__init__()

        # Encoder
        self.enc1 = ConvolutionBlock_Layer(input_channels, base_channels, t_embedding_dim)
        self.enc2 = ConvolutionBlock_Layer(base_channels, base_channels * 2, t_embedding_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvolutionBlock_Layer(base_channels * 2, base_channels * 4, t_embedding_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvolutionBlock_Layer(base_channels * 4, base_channels * 2, t_embedding_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvolutionBlock_Layer(base_channels * 2, base_channels, t_embedding_dim)

        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x, t_emb):
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        b = self.bottleneck(self.pool(e2), t_emb)

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1), t_emb)

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1), t_emb)

        return self.out_conv(d2)
    
class FlowNeuralNetwork(nn.Module):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 output_shape=(10,),
                 t_embedding_dim=128,
                 t_embedding_hidden_size=64,
                 unet_base_channels=64):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.t_embedding_dim = t_embedding_dim
        self.hidden_size = t_embedding_hidden_size

        self.time_embedding = TimeEmbedding_NeuralNetwork(
            input_size=1,
            hidden_size=t_embedding_hidden_size,
            output_size=t_embedding_dim,
            t_embedding_dim=t_embedding_dim
        )

        input_channels = input_shape[0]
        output_channels = output_shape[0]

        self.unet = UNet(
            input_channels=input_channels,
            base_channels=unet_base_channels,
            output_channels=output_channels,
            t_embedding_dim=t_embedding_dim
        )

    def forward(self, x, t):
        """
        Forward pass of the Flow Neural Network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step tensor of shape (B,).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_channels, H, W).
        """
        t_emb = self.time_embedding(t)  # shape: (B, t_embedding_dim)
        return self.unet(x, t_emb)