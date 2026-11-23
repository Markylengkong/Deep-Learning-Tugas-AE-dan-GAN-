import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden1=256, latent_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)
        self.relu = nn.ReLU()

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h1))
        return z

    def forward(self, x):
        z = self.encode(x)
        logits = self.fc3(z)
        return logits, z

    @torch.no_grad()
    def top_down_from_class(self, y_onehot):
        """Tahap 2: Proyeksi balik bobot dari vektor one-hot kelas y_c."""
        W1, W2, W3 = self.fc1.weight, self.fc2.weight, self.fc3.weight
        z_hat = torch.relu(y_onehot @ W3)
        h1_hat = torch.relu(z_hat @ W2)
        x_hat = torch.sigmoid(h1_hat @ W1)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def transpose_from_latent(self, z_mlp):
        """Tahap 3: Proyeksi balik langsung dari fitur laten z_MLP."""
        W1, W2 = self.fc1.weight, self.fc2.weight
        h1_hat = torch.relu(z_mlp @ W2)
        x_hat = torch.sigmoid(h1_hat @ W1)
        return x_hat.view(-1, 1, 28, 28)

class FrozenLatentDecoder(nn.Module):
    """Tahap 4: Decoder MLP terpisah (64 -> 256 -> 784) untuk z_MLP beku."""
    def __init__(self, latent_dim=64, hidden1=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28) 