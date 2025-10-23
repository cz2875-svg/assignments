import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)                          
        x = x.view(B, 128, 7, 7)               
        x = self.net(x)                        
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)  

    def forward(self, img):
        x = self.conv(img)                    
        x = x.view(x.size(0), -1)            
        logit = self.fc(x)                  
        return logit                      

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_dim = 100
    lr = 2e-4
    beta1 = 0.5
    batch_size = 64

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    criterion = nn.BCEWithLogitsLoss()  
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(16, z_dim, device=device)

    real_images = torch.randn(batch_size, 1, 28, 28, device=device)  
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # Train Discriminator
    D.zero_grad()
    # Real
    logits_real = D(real_images)                   
    loss_real = criterion(logits_real, real_labels)

    # Fake
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z).detach()                   
    logits_fake = D(fake_images)
    loss_fake = criterion(logits_fake, fake_labels)

    loss_D = (loss_real + loss_fake) * 0.5
    loss_D.backward()
    opt_D.step()
   
    # Train Generator
    G.zero_grad()
    z = torch.randn(batch_size, z_dim, device=device)
    gen_images = G(z)
    logits_gen = D(gen_images)                   
   
    loss_G = criterion(logits_gen, real_labels)
    loss_G.backward()
    opt_G.step()

    # Convert G output for visualization
    with torch.no_grad():
        sample = G(fixed_noise)                  
        sample_vis = (sample + 1.0) / 2.0         

    print("Single-step done. loss_D:", loss_D.item(), "loss_G:", loss_G.item())
