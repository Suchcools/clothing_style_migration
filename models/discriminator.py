import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=4,
                padding=1,
                bias=True,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)

        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[32, 64,128,256]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=4,
                padding=1,
                stride=2,
                padding_mode="reflect"

            ),
            nn.LeakyReLU(0.2),

        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels, out_channels=feature, stride=1 if feature==features[-1] else 2))
            in_channels=feature

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

# def test():
#     x = torch.randn((1, 3, 600, 400))
#     model = Discriminator(in_channels=3)
#     r = model(x)
#     print(r.size())

# if __name__ == '__main__':
#     test()

