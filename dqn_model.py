import torch
import torch.nn as nn

import torch.nn as nn

import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=27648, action_dim=3):  # Match saved model
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),  # net.0
            nn.ReLU(),                  # net.1
            nn.Linear(256, 128),        # net.2
            nn.ReLU(),                  # net.3
            nn.Linear(128, action_dim)  # net.4
        )

    def forward(self, x):
        return self.net(x)

