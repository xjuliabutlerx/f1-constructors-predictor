from torch import nn

class F1ConstructorsClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(F1ConstructorsClassifier, self).__init__()
        self.layer = nn.Sequential(
            ## Layer 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),

            # Layer 3
            nn.Linear(32, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)
    