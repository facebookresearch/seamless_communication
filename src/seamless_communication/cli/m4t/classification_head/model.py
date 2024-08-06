import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, n_layers: int = 3, n_classes: int = 5, n_heads: int = 16):
        super(ClassificationHead, self).__init__()
        self.num_languages = n_classes
        self.num_layers = n_layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
            + [nn.Linear(embed_dim, n_classes)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Seq, Embed)
        x = x[:, 0]
        for layer in self.layers:
            x = layer(x)
        return x
