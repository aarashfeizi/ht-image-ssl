import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, cfg) -> None:
        """
            cfg.emb_model.input_size: size of the input channels (int)
            cfg.emb_model.sizes: size of the channels for the encoder and decoder (list of ints)
        """
        super().__init__()
        encoder_layers = []
        dencoder_layers = []
        channel_sizes = [cfg.emb_model.input_size] # channel sizes
        channel_sizes.extend(cfg.emb_model.sizes) # channel sizes
        kernel_sizes = cfg.emb_model.kernel_sizes # kernel sizes
        assert len(cfg.emb_model.sizes) == len(cfg.emb_model.kernel_sizes)
        self.is_train = False
        for idx in range(1, len(channel_sizes)):
            layer = nn.Conv2d(in_channels=channel_sizes[idx - 1], out_channels=channel_sizes[idx], kernel_size=kernel_sizes[idx - 1])
            relu = nn.ReLU()
            maxpool = nn.MaxPool2d()
            encoder_layers.extend([layer, relu, maxpool])
        
        self.avgpool = nn.AvgPool2d()

        for idx in range(len(channel_sizes) - 1, 0, -1):
            layer = nn.ConvTranspose2d(in_channels=channel_sizes[idx], out_channels=channel_sizes[idx - 1], kernel_size=kernel_sizes[idx - 1])
            relu = nn.ReLU()
            maxunpool = nn.MaxUnpool2d()
            dencoder_layers.extend([layer, relu, maxunpool])
        

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*dencoder_layers)
    
    def train(self):
        super()
        self.is_train = True
    
    def eval(self):
        super()
        self.is_train = False

    def forward(self, x):
        latent = self.encoder(x)
        if self.is_train:
            x_pred = self.decoder(latent)
            return x_pred
        else:
            latent = self.avgpool(latent)
            assert len(latent.shape) == 2
            return latent