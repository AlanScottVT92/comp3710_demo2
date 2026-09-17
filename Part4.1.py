import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

      
        # normalization

        self.conv = nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        # activation
        

    def forward(self, x):
        # Conv -> Normalization -> Activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.block1 = EncoderBlock(1, 32)
        self.block2 = EncoderBlock(32, 64)
        self.block3 = EncoderBlock(64, 128)
        self.block4 = EncoderBlock(128, 256)
        self.block5 = EncoderBlock(256, 512)

        # Flatten
        self.flatten = nn.Flatten()

        # feature -> mu
        self.mu = nn.Linear(512 * 8 * 8, latent_dim)
        # feature -> logvar
        self.logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_channels)
        
        # normalization

        self.activation = nn.LeakyReLU(0.2)
        # activation

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.decoder_input = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decode1 = DecoderBlock(512, 256)
        self.decode2 = DecoderBlock(256, 128)
        self.decode3 = DecoderBlock(128, 64)
        self.decode4 = DecoderBlock(64, 32)

        self.output = nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=1,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 8, 8)

        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

       

    def reparameterize(self, mu, logvar):
        # 在这里计算 z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        print("z shape:", z.shape)
        return reconstructed, mu, logvar




if __name__ == "__main__":
    latent_dim = 64

    model = VAE(latent_dim)

    # 模拟一个 batch：4 张单通道 256×256 MRI
    x = torch.randn(4, 1, 256, 256)

    reconstructed, mu, logvar = model(x)

    print("Input shape:        ", x.shape)
    print("Mu shape:           ", mu.shape)
    print("Logvar shape:       ", logvar.shape)
    print("Reconstructed shape:", reconstructed.shape)
    print("Reconstructed min:", reconstructed.min().item())
    print("Reconstructed max:", reconstructed.max().item())