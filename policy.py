
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
  

    def __init__(self, n_res_blocks: int = 10, channels: int = 128):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(13, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(n_res_blocks)]
        )

    
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),   
        )

    def _flat_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
    
        B = x.shape[0]
        pieces = x[:, :768].view(B, 12, 8, 8)          
        turn   = x[:, 768].view(B, 1, 1, 1).expand(B, 1, 8, 8)  
        return torch.cat([pieces, turn], dim=1)         

    def forward(self, x: torch.Tensor):
        """
        x       : (B, 769)
        returns : policy_logits (B, 4096), value (B, 1)
        """
        spatial = self._flat_to_spatial(x)              # (B, 13, 8, 8)
        trunk   = self.res_tower(self.input_conv(spatial))
        return self.policy_head(trunk), self.value_head(trunk)

    def inference(self, x: torch.Tensor, legal_mask: torch.Tensor):
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x.unsqueeze(0))
            logits[0][~legal_mask] = float('-inf')
            priors = F.softmax(logits[0], dim=0)
        return priors, value.item()

