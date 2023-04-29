# /**
#  * @author shaowinw & GPT
#  * @email shaowinw@geophyai.com
#  * @create date 2023-04-29 17:34:55
#  * @modify date 2023-04-29 17:34:55
#  * @desc [Autoencoder for denoising <supervised>]
#  */

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for data, _ in train_loader:
        noisy_data = data + torch.randn(*data.shape) * 0.5
        noisy_data = torch.clamp(noisy_data, 0., 1.)
        
        noisy_data = noisy_data.to(device)
        target = data.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

import matplotlib.pyplot as plt

# Function to display images
def imshow(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

# Test the trained model
model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        noisy_data = data + torch.randn(*data.shape) * 0.5
        noisy_data = torch.clamp(noisy_data, 0., 1.)

        noisy_data = noisy_data.to(device)
        denoised_data = model(noisy_data)

        # Move data back to CPU for visualization
        noisy_data = noisy_data.cpu()
        denoised_data = denoised_data.cpu()

        # Visualize the results
        for i in range(5):
            noisy_image = noisy_data[i].squeeze().numpy()
            denoised_image = denoised_data[i].squeeze().numpy()
            original_image = data[i].squeeze().numpy()

            imshow(original_image, f"Original Image {i + 1}")
            imshow(noisy_image, f"Noisy Image {i + 1}")
            imshow(denoised_image, f"Denoised Image {i + 1}")

        # We only visualize the first batch
        break