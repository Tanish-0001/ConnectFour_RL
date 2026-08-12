import torch
import torch.nn as nn


class DQNNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU()
        )

        input_size = 6 * 7 * 64

        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7)
        )

    def forward(self, x):
        x = self.cnn(x)
        # [batch_size, 64, 6, 7] -> [batch_size, 64 * 6 * 7]
        x = torch.flatten(x, start_dim=1)
        return self.linear_layers(x)


if __name__ == "__main__":
    model = DQNNet()
    total = sum(p.numel() for p in model.parameters())
    print(total)

    inp = torch.randn((32, 2, 6, 7))
    out = model(inp)
    print(out.shape)
