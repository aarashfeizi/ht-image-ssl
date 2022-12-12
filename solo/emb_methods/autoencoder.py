import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_size, sizes=[256, 128, 64]) -> None:
        super().__init__()
        encoder_layers = []
        dencoder_layers = []
        layers_sizes = [input_size]
        layers_sizes.extend(sizes)
        for idx in range(1, len(layers_sizes)):
            layer = nn.Linear(layers_sizes[idx - 1], layers_sizes[idx])
            relu = nn.ReLU()
            encoder_layers.extend([layer, relu])

        encoder_layers = encoder_layers[:-1] # remove last relu from encoder

        for idx in range(len(layers_sizes) - 1, 0, -1):
            layer = nn.Linear(layers_sizes[idx - 1], layers_sizes[idx])
            relu = nn.ReLU()
            dencoder_layers.extend([layer, relu])
        
        dencoder_layers = dencoder_layers[:-1] # remove last relu from dencoder

        self.encoder = nn.Sequential(encoder_layers)
        self.decoder = nn.Sequential(dencoder_layers)

    
    def forward(self, x):
        latent = self.encoder(x)
        x_pred = self.decoder(latent)
        return x_pred