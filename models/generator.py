import torch
import torch.nn as nn

## 残差网络模型

# 卷积结构
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# 残差结构
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return  x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features = 64, num_residuals=9):
        super(Generator, self).__init__()
        # (batch_size, img_channels, 256, 256)  ---->  (batch_size, num_features, 256, 256)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        # (batch_size, num_features, 256, 256)  ---->  (batch_size, num_features*2, 128, 128)
        # (batch_size, num_features*2, 128, 128)  ----> (batch_size, num_features*4, 64, 64)
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels=num_features, out_channels=num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_channels=num_features*2, out_channels=num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        # (batch_size, num_features*4, 64, 64) ---->  (batch_size, num_features*2, 128, 128)
        # (batch_size, num_features * 2, 128, 128) ----> (batch_size, num_features, 256, 256)
        self.up_blcoks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        # (batch_size, num_features, 256, 256) ----> (batch_size, 3, 256, 256)
        self.last = nn.Conv2d(in_channels=num_features*1, out_channels=img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_block(x)
        for layer in self.up_blcoks:
            x = layer(x)

        return torch.tanh(self.last(x))


# def test():
#     x = torch.randn((1, 3, 600, 400))
#     model = Generator(img_channels=3)
#     r = model(x)
#     print(r.size())

# if __name__ == '__main__':
#     test()
