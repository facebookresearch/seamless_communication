import torch
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_languages, hidden_dim, num_layers):
        super(ClassificationHead, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, num_languages)

    def forward(self, x):
        '''
        Returns probabilities of each language
        '''
        for layer in self.layers:
            x = layer(x)
        return torch.softmax(self.output_layer(x), dim=-1)
    