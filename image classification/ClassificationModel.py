from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.channel_size = 512
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, self.channel_size),
            nn.ReLU(),
            nn.Linear(self.channel_size, self.channel_size),
            nn.ReLU(),
            nn.Linear(self.channel_size, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)
