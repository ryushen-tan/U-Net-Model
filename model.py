import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Double Convolutional Layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 1st Convolutional Layer
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # Batch Normalization
            nn.BatchNorm2d(out_channels),
            # Activation Function
            nn.ReLU(inplace=True),
            # 2nd Convolutional Layer
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # Batch Normalization
            nn.BatchNorm2d(out_channels),
            # Activation Function
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):
        return self.conv(x)
    
# UNET Architecture
class UNET(nn.Module):
    def __init__(
            # UNET Architecture
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            # Maps 3 to 64 channels
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                # Transpose Convolution with feature*2 because of concatenation
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            # Double Convolution
            self.ups.append(DoubleConv(feature*2, feature))

            # Maps 512 to 256 channels
        self.bottle_neck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # Forward Pass
    def forward(self, x):
        skip_connections = []
        # Down part of UNET
        for down in self.downs:
            x = down(x)
            # Append to skip_connections
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for idx in range (0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)

            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()