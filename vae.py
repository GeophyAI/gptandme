# /**
#  * @author shaowinw
#  * @email shaowinw@geophyai.com
#  * @create date 2023-04-29 18:09:08
#  * @modify date 2023-04-29 18:09:08
#  * @desc [VAE for denoising <unsupervised>]
#  */

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU()
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.view(data.size(0), -1)
        noisy_data = data + torch.randn(*data.shape) * 0.5
        noisy_data = torch.clamp(noisy_data, 0., 1.).to(device)

        optimizer.zero_grad()
        recon_data, mu, logvar = model(noisy_data)
        loss = vae_loss(recon_data, noisy_data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset)}")

def plot_images(images, noisy_images, denoised_images):
    fig, axes = plt.subplots(3, len(images), figsize=(15, 5))
    for i, (image, noisy_image, denoised_image) in enumerate(zip(images, noisy_images, denoised_images)):
        axes[0, i].imshow(image, cmap='gray')
        axes[1, i].imshow(noisy_image, cmap='gray')
        axes[2, i].imshow(denoised_image, cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')
    plt.show()

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)

# Get a batch of test data
images, _ = next(iter(test_loader))
images = images[:10].view(-1, 28 * 28)
noisy_images = images + torch.randn(*images.shape) * 0.5
noisy_images = torch.clamp(noisy_images, 0., 1.)

with torch.no_grad():
    model.eval()
    denoised_images, _, _ = model(noisy_images.to(device))
    denoised_images = denoised_images.cpu()

plot_images(images.view(-1, 28, 28), noisy_images.view(-1, 28, 28), denoised_images.view(-1, 28, 28))
