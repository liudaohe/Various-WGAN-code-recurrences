
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockDiscriminator_gp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径: [3×3]×2 + Down
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 第一个3×3
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # 第二个3×3
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)  # Down
        )

        # 旁路下采样
        self.bypass = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.bypass(x)




class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径: [3×3]×2 + Down
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 第一个3×3
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # 第二个3×3
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False),  # Down
            nn.BatchNorm2d(out_channels)
        )

        # 旁路下采样
        self.bypass = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main(x) + self.bypass(x)


class Discriminator(nn.Module):
    def __init__(self,judge=True,sig = False):
        super().__init__()
        # 初始3×3卷积
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.sig = sig
        # 残差块配置
        channels = [
            (64, 128),  # 64×64 -> 32×32
            (128, 256),  # 32×32 -> 16×16
            (256, 512),  # 16×16 -> 8×8
            (512, 512)  # 8×8 -> 4×4
        ]
        
        if judge:
            self.res_blocks = nn.ModuleList([
                ResBlockDiscriminator(in_ch, out_ch) for in_ch, out_ch in channels
            ])
        else:
            self.res_blocks = nn.ModuleList([
                ResBlockDiscriminator_gp(in_ch, out_ch) for in_ch, out_ch in channels
            ])

        # 输出层
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # 初始卷积
        x = F.relu(self.conv1(x))

        # 通过残差块
        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.relu(x)
        # 展平并输出
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc(x)
        if self.sig:
            return torch.sigmoid(x.squeeze())
        else:
            return x.squeeze()
