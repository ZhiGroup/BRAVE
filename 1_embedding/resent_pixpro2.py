
import torch
import torch.nn as nn
import lightning.pytorch as pl


class UNet3D(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = self._block(in_channels, base_channels)
        self.down1 = nn.Conv3d(base_channels, base_channels,
                               kernel_size=3, stride=2, padding=1)

        self.enc2 = self._block(base_channels, base_channels * 2)
        self.down2 = nn.Conv3d(
            base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.down3 = nn.Conv3d(
            base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.enc4 = self._block(base_channels * 4, base_channels * 8)
        self.down4 = nn.Conv3d(
            base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self._block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels*2)

        # Final Convolution
        self.final_conv = nn.Conv3d(
            base_channels*2, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        '''
            x: (b,1, 128,128,128)
        '''
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        # Bottleneck
        b = self.bottleneck(d4)  # (1, 1024, 8, 8, 8) #(1, 1024, 6, 6, 6)

        # Decoder
        u4 = self.up4(b)  # (1, 512, 16, 16, 16) #(1, 512, 12, 12, 12)
        concat4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(concat4)

        u3 = self.up3(d4)  # (1, 256, 32, 32, 32)
        concat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(concat3)  # (1, 256, 32, 32, 32) #(1, 512, 24, 24, 24)

        u2 = self.up2(d3) #(1, 512, 48, 48, 48)
        concat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(concat2)  # (1, 256, 64, 64, 64) #(1, 512, 48, 48, 48)

        u1 = self.up1(d2)  # (1, 64, 128, 128, 128) # (1, 512, 96, 96, 96)
        concat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(concat1)  # (1, 128, 128, 128, 128)

        # Final Output
        out = self.final_conv(d1)  # (1, 1, 128, 128, 128)

        return out,  d1,d3

# # Instantiate the model
# model = UNet3D(in_channels=1, out_channels=1)

# # Test with a random tensor
# x = torch.randn(1, 1, 96, 96, 96)  # [Batch, Channels, Depth, Height, Width]

# output = model(x)
