# model.py - Defines the Pix2Pix Generator and Discriminator neural networks

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    Generator network that converts RGB images to NDVI images.
    Uses U-Net architecture: encoder (downsamples) -> decoder (upsamples)
    Skip connections help preserve spatial information.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the generator network.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for NDVI as RGB)
        """
        super(UNetGenerator, self).__init__()
        
        # ENCODER: Progressively downsample the image and extract features
        # Each block reduces spatial size by 2x and increases feature channels
        
        self.down1 = self.down_block(in_channels, 64, normalize=False)  # 256->128
        self.down2 = self.down_block(64, 128)   # 128->64
        self.down3 = self.down_block(128, 256)  # 64->32
        self.down4 = self.down_block(256, 512)  # 32->16
        self.down5 = self.down_block(512, 512)  # 16->8
        self.down6 = self.down_block(512, 512)  # 8->4
        
        # BOTTLENECK: Deepest layer with most compressed representation
        self.bottleneck = self.down_block(512, 512)  # 4->2
        
        # DECODER: Progressively upsample back to original size
        # Skip connections concatenate encoder features with decoder features
        
        self.up1 = self.up_block(512, 512, dropout=True)   # 2->4
        self.up2 = self.up_block(1024, 512, dropout=True)  # 4->8 (1024=512+512 from skip)
        self.up3 = self.up_block(1024, 512, dropout=True)  # 8->16
        self.up4 = self.up_block(1024, 256)  # 16->32
        self.up5 = self.up_block(512, 128)   # 32->64
        self.up6 = self.up_block(256, 64)    # 64->128
        
        # FINAL LAYER: Convert features back to RGB image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )
    
    def down_block(self, in_channels, out_channels, normalize=True):
        """
        Create a downsampling block: Conv -> BatchNorm -> LeakyReLU
        Reduces spatial dimensions by 2x.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))  # Normalize activations
        layers.append(nn.LeakyReLU(0.2))  # Activation function
        return nn.Sequential(*layers)
    
    def up_block(self, in_channels, out_channels, dropout=False):
        """
        Create an upsampling block: TransposeConv -> BatchNorm -> ReLU
        Increases spatial dimensions by 2x.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))  # Randomly drop 50% of neurons (prevents overfitting)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass: process input through encoder, bottleneck, and decoder.
        
        Args:
            x: Input RGB image tensor
            
        Returns:
            Generated NDVI image tensor
        """
        # ENCODER: Save outputs for skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # BOTTLENECK
        bottleneck = self.bottleneck(d6)
        
        # DECODER: Concatenate skip connections from encoder
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], dim=1))  # Concatenate along channel dimension
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        
        # FINAL OUTPUT
        return self.final(torch.cat([u6, d1], dim=1))


class PatchGANDiscriminator(nn.Module):
    """
    Discriminator network that judges if an image pair is real or fake.
    Uses PatchGAN: classifies each 70x70 patch as real/fake instead of whole image.
    This helps generate sharper, more detailed images.
    """
    
    def __init__(self, in_channels=6):
        """
        Initialize the discriminator.
        
        Args:
            in_channels: 6 channels (3 for RGB input + 3 for NDVI output)
        """
        super(PatchGANDiscriminator, self).__init__()
        
        # Stack of convolutional layers that progressively downsample
        self.model = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 32x32 -> 31x31 (stride=1 to maintain resolution)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Final layer: 31x31 -> 30x30, outputs single channel (real/fake score)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # Output probability between 0 (fake) and 1 (real)
        )
    
    def forward(self, x, y):
        """
        Forward pass: concatenate input and output images, then classify.
        
        Args:
            x: Input RGB image
            y: Output NDVI image (real or generated)
            
        Returns:
            Probability map indicating real/fake for each patch
        """
        # Concatenate RGB and NDVI along channel dimension (3+3=6 channels)
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)
