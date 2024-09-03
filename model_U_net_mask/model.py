import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

# Example usage:
# model = UNet3D(in_channels=1, out_channels=1, init_features=32)
# input_tensor = torch.randn(1, 1, 128, 128, 128)  # Batch size of 1, 1 input channel, 128x128x128 volume
# output = model(input_tensor)
# print(output.shape)  # Should print torch.Size([1, 1, 128, 128, 128])

if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    input_tensor = torch.randn(1, 1, 224, 224, 224)  # Batch size of 1, 1 input channel, 224x224x224 volume
    output = model(input_tensor)
    print(output.shape)  # Should print torch.Size([1, 1, 224, 224, 224])