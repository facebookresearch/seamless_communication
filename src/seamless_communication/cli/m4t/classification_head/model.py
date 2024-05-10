from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, n_layers, n_classes):
        super(ClassificationHead, self).__init__()
        self.num_languages = n_classes
        self.num_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = self.input_dim
        
        self.layers = nn.ModuleList(
            [ nn.Linear(input_dim, input_dim) for _ in range(n_layers) ] + \
            [ nn.Linear(input_dim, n_classes) ])

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return nn.functional.softmax(x)
    