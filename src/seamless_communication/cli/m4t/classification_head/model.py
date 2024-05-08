import torch
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, num_languages, num_layers):
        super(ClassificationHead, self).__init__()
        self.num_languages = num_languages
        self.num_layers = num_layers
        self.hidden_dim = None
        self.input_dim = None
        self.layers = None

    def forward(self, x):
        if self.layers is None:
            self.input_dim = x.size(-1)
            self.layers = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_languages)
            )
        return self.layers(x)
    