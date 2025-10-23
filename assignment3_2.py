import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import os

# Define Generator and Discriminator (same as before)
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),    
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, img):
        x = self.net(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
batch_size = 128
epochs = 30
lr = 2e-4
beta1 = 0.5

os.makedirs("gan_images", exist_ok=True)

# MNIST dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

G = Generator(z_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init_normal)
D.apply(weights_init_normal)

criterion = nn.BCEWithLogitsLoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, z_dim, device=device)

# Training Loop
for epoch in range(1, epochs + 1):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z).detach()

        real_loss = criterion(D(real_imgs), real)
        fake_loss = criterion(D(fake_imgs), fake)
        loss_D = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(batch_size, z_dim, device=device)
        gen_imgs = G(z)
        loss_G = criterion(D(gen_imgs), real)  

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if i % 200 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {i}/{len(train_loader)} "
                f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
            )

    # Save sample grid each epoch
    with torch.no_grad():
        fake_samples = G(fixed_noise)
        fake_samples = (fake_samples + 1) / 2.0  
        grid = make_grid(fake_samples, nrow=8)
        save_image(grid, f"gan_images/epoch_{epoch:03d}.png")

print("Training complete! Check the gan_images/ folder for results.")

from IPython.display import Image, display
display(Image(filename="gan_images/epoch_010.png"))