import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
data_dir = r"D:\academic\Semester 2 2026\COMP3710\Demo2\Part4\keras_png_slices_data\keras_png_slices_data"

train_dir = os.path.join(data_dir, "keras_png_slices_train")
val_dir = os.path.join(data_dir, "keras_png_slices_validate")
test_dir = os.path.join(data_dir, "keras_png_slices_test")

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

      
        # normalization

        self.conv = nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        # activation
        

    def forward(self, x):
        # Conv -> Normalization -> Activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.block1 = EncoderBlock(1, 32)
        self.block2 = EncoderBlock(32, 64)
        self.block3 = EncoderBlock(64, 128)
        self.block4 = EncoderBlock(128, 256)
        self.block5 = EncoderBlock(256, 512)

        # Flatten
        self.flatten = nn.Flatten()

        # feature -> mu
        self.mu = nn.Linear(512 * 8 * 8, latent_dim)
        # feature -> logvar
        self.logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_channels)
        
        # normalization

        self.activation = nn.LeakyReLU(0.2)
        # activation

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.decoder_input = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decode1 = DecoderBlock(512, 256)
        self.decode2 = DecoderBlock(256, 128)
        self.decode3 = DecoderBlock(128, 64)
        self.decode4 = DecoderBlock(64, 32)

        self.output = nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=1,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 8, 8)

        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

       

    def reparameterize(self, mu, logvar):
        # 在这里计算 z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar

def vae_loss(reconstructed, x, mu, logvar, beta):
    # Reconstruction loss
    recon_loss = F.mse_loss(
        reconstructed,
        x,
        reduction="mean"
    )

    # KL divergence
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )

    # Average across the batch
    kl_loss = kl_loss.mean()

    # Total VAE loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

class OASISDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):       #The number of samples
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    print("Device:", device)
    
    train_dataset = OASISDataset(train_dir, transform)
    val_dataset = OASISDataset(val_dir, transform)
    test_dataset = OASISDataset(test_dir, transform)

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    images = next(iter(train_loader))

    print("Batch shape:", images.shape)
    print("Batch min:", images.min().item())
    print("Batch max:", images.max().item())

    epochs=20
    latent_dim = 64
    beta = 0.0001

    model = VAE(latent_dim=64).to(device)

    images = images.to(device)

    reconstructed, mu, logvar = model(images)

    print("Reconstructed:", reconstructed.shape)
    print("Mu:", mu.shape)
    print("Logvar:", logvar.shape)

    total_loss, recon_loss, kl_loss = vae_loss(
        reconstructed,
        images,
        mu,
        logvar,
        beta
    )

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
    )

    print("Reconstruction loss:", recon_loss.item())
    print("KL loss:", kl_loss.item())
    print("Total loss:", total_loss.item())

    print("beta =", beta)
    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0
        max_abs_mu = 0.0
        max_logvar = float("-inf")
        min_logvar = float("inf")
      
        for images in train_loader:
            # 1. Move batch to GPU
            images = images.to(device)

            # 2. Clear old gradients
            optimizer.zero_grad()

            # 3. Forward
            reconstructed, mu, logvar = model(images)

            # 4. Calculate VAE loss
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstructed,
                images,
                mu,
                logvar,
                beta
            )

            # 5. Backpropagation
            total_loss.backward()

            # 6. Update parameters
            optimizer.step()

            running_total += total_loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()

            max_abs_mu = max(
                max_abs_mu,
                mu.detach().abs().max().item()
            )

            max_logvar = max(
                max_logvar,
                logvar.detach().max().item()
            )

            min_logvar = min(
                min_logvar,
                logvar.detach().min().item()
            )
        num_batches = len(train_loader)

        avg_total = running_total / num_batches
        avg_recon = running_recon / num_batches
        avg_kl = running_kl / num_batches

        # =========================
        # Validation
        # =========================
        model.eval()

        val_running_total = 0.0
        val_running_recon = 0.0
        val_running_kl = 0.0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)

                reconstructed, mu, logvar = model(images)

                total_loss, recon_loss, kl_loss = vae_loss(
                    reconstructed,
                    images,
                    mu,
                    logvar,
                    beta
                )

                val_running_total += total_loss.item()
                val_running_recon += recon_loss.item()
                val_running_kl += kl_loss.item()

        num_val_batches = len(val_loader)

        avg_val_total = val_running_total / num_val_batches
        avg_val_recon = val_running_recon / num_val_batches
        avg_val_kl = val_running_kl / num_val_batches

        print(
            f"Epoch {epoch + 1}, "
            f"Train Recon: {avg_recon:.4f}, "
            f"Train KL: {avg_kl:.4f}, "
            f"Train Total: {avg_total:.4f}"
        )

        print(
            f"          "
            f"Val Recon: {avg_val_recon:.4f}, "
            f"Val KL: {avg_val_kl:.4f}, "
            f"Val Total: {avg_val_total:.4f}"
        )

        print(
            f"          "
            f"max|mu|: {max_abs_mu:.2f}, "
            f"logvar range: [{min_logvar:.2f}, {max_logvar:.2f}]"
        )

    model.eval()

    # Get one batch from validation set
    images = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        reconstructed, mu, logvar = model(images)

    # Move tensors back to CPU for plotting
    images = images.cpu()
    reconstructed = reconstructed.cpu()

    num_images = 6

    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Original image
        axes[0, i].imshow(images[i, 0], cmap="gray")
        axes[0, i].axis("off")

        # Reconstructed image
        axes[1, i].imshow(reconstructed[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)

    plt.tight_layout()
    plt.show()


    # =========================
    # Generate new MRI samples
    # =========================

    model.eval()

    num_samples = 12

    with torch.no_grad():
        # Sample latent vectors from N(0, I)
        z = torch.randn(
            num_samples,
            latent_dim,
            device=device
        )

        # Decode latent vectors into MRI images
        generated_images = model.decoder(z)

    # Move to CPU for matplotlib
    generated_images = generated_images.cpu()

    # Plot generated images
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))

    for i in range(num_samples):
        row = i // 6
        col = i % 6

        axes[row, col].imshow(
            generated_images[i, 0],
            cmap="gray"
        )
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


    # =========================
    # Collect latent representations for UMAP
    # =========================

    model.eval()

    all_mu = []

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)

            mu, logvar = model.encoder(images)

            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)

    print("Latent representation shape:", all_mu.shape)
    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    embedding = reducer.fit_transform(
        all_mu.numpy()
    )

    print("UMAP shape:", embedding.shape)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=8,
        alpha=0.6
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("VAE Latent Space - OASIS Validation Set")

    plt.show()

    # =========================
    # Analyse UMAP by case
    # =========================

    case_ids = []

    for filename in val_dataset.image_files:
        parts = filename.split("_")
        case_id = parts[1]
        case_ids.append(case_id)

    print("Number of images:", len(case_ids))
    print("First 20 case IDs:", case_ids[:20])
    print("Number of cases:", len(set(case_ids)))

    # Verify that every UMAP point has one corresponding case ID
    print("Number of case IDs:", len(case_ids))
    print("Number of UMAP points:", embedding.shape[0])
    print(
        "Do case IDs match UMAP points?",
        len(case_ids) == embedding.shape[0]
    )

    # Convert case IDs into integer labels
    unique_cases = sorted(set(case_ids))

    case_to_number = {
    case_id: i
    for i, case_id in enumerate(unique_cases)
    }

    case_numbers = [
    case_to_number[case_id]
    for case_id in case_ids
    ]

    # Plot UMAP coloured by case
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=case_numbers,
    s=10,
    alpha=0.8,
    cmap="tab20"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("VAE Latent Space - Coloured by Case")

    plt.colorbar(scatter, label="Case index")

    plt.show()

    # =========================
    # Analyse UMAP by slice number
    # =========================

    slice_numbers = []

    for filename in val_dataset.image_files:
        parts = filename.split("_")
        slice_number = int(parts[3].split(".")[0])
        slice_numbers.append(slice_number)

    # Check extracted slice numbers
    print("Number of slice numbers:", len(slice_numbers))
    print("First 20 slice numbers:", slice_numbers[:20])
    print(
        "Slice number range:",
        min(slice_numbers),
        "to",
        max(slice_numbers)
    )

    print(
        "Do slice numbers match UMAP points?",
        len(slice_numbers) == embedding.shape[0]
    )

    # Plot UMAP coloured by slice number
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=slice_numbers,
        s=10,
        alpha=0.8,
        cmap="viridis"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("VAE Latent Space - Coloured by Slice Number")

    plt.colorbar(scatter, label="Slice number")

    plt.show()
   

if __name__ == "__main__":
    main()