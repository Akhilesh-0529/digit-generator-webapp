

# Training Script
from model import DigitGenerator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_classes = 10
lr = 0.0002
batch_size = 64
epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DigitGenerator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch_idx, (real_imgs, labels) in enumerate(train_loader):
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, latent_dim).to(device)
        one_hot = torch.zeros(batch_size, num_classes).to(device)
        one_hot[range(batch_size), labels] = 1
        target_imgs = real_imgs.view(batch_size, -1).to(device)
        generated_imgs = model(z, one_hot)
        loss = criterion(generated_imgs.view(batch_size, -1), target_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
