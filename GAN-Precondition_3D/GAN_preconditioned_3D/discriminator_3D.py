import torch
import torch.nn as nn
class PatchGAN3D(nn.Module):
    def __init__(self):
        super(PatchGAN3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.Tanh(),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.Tanh(),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.Tanh(),

            nn.Conv3d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.Tanh(),

            nn.Conv3d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # 在通道维度拼接
        x = self.model(x)
        return x.view(x.size(0), -1)  # 展平输出
