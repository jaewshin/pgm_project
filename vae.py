from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        self.input_dims = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.fc4 = nn.Linear(hidden_dims[1], latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to("mps") # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to("mps")
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = self.fc4(x)
        z = mu + sigma * self.N.sample(mu.shape)

        self.mu = mu 
        self.sigma = sigma  
        self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

        return z

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        self.input_dims = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.input_dims = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        print("input_dim: ", input_dim)
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dims, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
            
def train(VAE, data, epochs=100):
    crossent_loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(VAE.parameters())
    losses = []
    for _ in range(epochs):
        data = data.to("mps")
        opt.zero_grad()
        x_hat = VAE(data)
        reconstruction_loss = crossent_loss(data, x_hat) # categorical crossentropy
        loss = (reconstruction_loss + VAE.encoder.kl).mean() # negative log likelihood = reconstruction loss + KL divergence
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return VAE, losses

def prob_new(VAE, data):
    '''Returns the probability of a new data point under the VAE model'''
    data = data.to("mps")
    crossent_loss = nn.CrossEntropyLoss()
    x_hat = VAE(data)
    reconstruction_loss = crossent_loss(data, x_hat) # categorical crossentropy

    mu, sigma = VAE.encoder.mu, VAE.encoder.sigma
    kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    loss = (reconstruction_loss + kl).mean() # negative log likelihood = reconstruction loss + KL divergence
    prob = torch.exp(-loss)
    return prob

