cat << 'EOF' > src/utils.py
import time
import random
import numpy as np
from pexpect import EOF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms, models
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import linalg

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(batch_size=128, seed=42, data_dir="./data"):
    transform = transforms.Compose([transforms.ToTensor()])
    train_raw = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_raw, test_raw])

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_reconstruction_metrics(orig_tensor, recon_tensor):
    orig = orig_tensor.detach().cpu().numpy()
    recon = np.clip(recon_tensor.detach().cpu().numpy(), 0.0, 1.0)

    mse = float(np.mean((orig - recon) ** 2))
    psnr_list, ssim_list = [], []
    for i in range(orig.shape[0]):
        o = orig[i, 0]
        r = recon[i, 0]
        psnr_list.append(peak_signal_noise_ratio(o, r, data_range=1.0))
        ssim_list.append(structural_similarity(o, r, data_range=1.0))
    return {
        "MSE": mse,
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
    }

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.model(x).cpu().numpy()

def calculate_fid(real_images, fake_images, device, batch_size=128):
    extractor = InceptionFeatureExtractor(device)

    def get_activations(imgs):
        acts = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size].to(device)
            acts.append(extractor(batch))
        return np.concatenate(acts, axis=0)

    act1 = get_activations(real_images)
    act2 = get_activations(fake_images)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

@torch.no_grad()
def measure_latency_ms(sample_fn, device, repeats=20):
    _ = sample_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = sample_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times))
EOF