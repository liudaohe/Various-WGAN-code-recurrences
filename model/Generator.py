
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockGenerator_gp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径: [3×3]×2 + Up
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 第一个3×3
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # 第二个3×3
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False) # Up
        )

        # 旁路上采样
        self.bypass = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.bypass(x)



class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径: [3×3]×2 + Up
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 第一个3×3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # 第二个3×3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False),  # Up
            nn.BatchNorm2d(out_channels)
        )

        # 旁路上采样
        self.bypass = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self,judge=True, z_dim=128):
        super().__init__()
        # 从噪声到4×4
        if judge:
            self.fc = nn.Sequential(
                nn.Linear(z_dim, 512 * 4 * 4, bias=False),
                nn.BatchNorm1d(512 * 4 * 4),
                nn.ReLU(True)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(z_dim, 512 * 4 * 4, bias=False),
                nn.ReLU(True)
            )
        # 残差块配置
        channels = [
            (512, 512),  # 4×4 -> 8×8
            (512, 256),  # 8×8 -> 16×16
            (256, 128),  # 16×16 -> 32×32
            (128, 64)  # 32×32 -> 64×64
        ]
        if judge:
            self.res_blocks = nn.ModuleList([
                ResBlockGenerator(in_ch, out_ch) for in_ch, out_ch in channels
            ])
        else:
            self.res_blocks = nn.ModuleList([
                ResBlockGenerator_gp(in_ch, out_ch) for in_ch, out_ch in channels
            ])
        self.relu = nn.ReLU(True)

        # 最终3×3卷积
        self.final = nn.Conv2d(64, 3, 3, 1, 1, bias=False)

    def forward(self, z):
        # 从噪声生成初始特征
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)

        # 通过残差块
        for res_block in self.res_blocks:
            x = res_block(x)

        # 最终输出
        x = self.relu(x)
        x = torch.tanh(self.final(x))
        return x
