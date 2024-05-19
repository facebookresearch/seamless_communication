from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_layers, n_classes, n_heads = 4):
        super(ClassificationHead, self).__init__()
        self.num_languages = n_classes
        self.num_layers = n_layers
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads)
        self.layers = nn.ModuleList(
            [ nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),  # normalize batch
                nn.ReLU(),  # activation function
                nn.Dropout(0.5)  # prevent overfitting
              ) for _ in range(n_layers)
            ] + [ nn.Linear(embed_dim, n_classes) ])

    def forward(self, x):
        # (Batch, Seq, Embed)
        x, _ = self.attn(x, x, x)
        x = x[:, 0]
        for layer in self.layers:
            x = layer(x)
        return nn.functional.softmax(x).float()
    