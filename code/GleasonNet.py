import torch.nn as nn
import torch.nn.functional as F


class GleasonNet(nn.Module):
    def __init__(self, input_features, num_class):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=256,
                kernel_size=2,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, 2, 1, 2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, 2, 1, 2),
            nn.ReLU(),
        )

        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # print(output4)
        self.fc = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_class),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, num_class)
        )

    def forward(self, x):
        # print(x)
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        # print(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten the output
        x = F.dropout(x, p=0.5)
        # print(x.shape)
        output = self.fc(x)
        return output
