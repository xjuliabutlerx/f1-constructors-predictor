from torch import nn

class F1ConstructorsClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(F1ConstructorsClassifier, self).__init__()
        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),

            # Layer 2
            nn.Linear(128, 64),

            # Layer 3
            nn.Linear(64, 32),

            # Layer 4
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)
    