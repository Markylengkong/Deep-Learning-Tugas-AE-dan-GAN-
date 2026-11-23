import time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.utils import (
    set_seed, get_device, get_dataloaders,
    compute_reconstruction_metrics, calculate_fid, measure_latency_ms
)
from models.autoencoders import VanillaAE, SparseAE, VAE
from models.gans import (
    VanillaGenerator, VanillaDiscriminator,
    DCGANGenerator, DCGANDiscriminator, weights_init_dcgan
)
from models.ddpm import DDPM

def save_grid(tensor_imgs, path, nrow=8, title=None):
    n = tensor_imgs.size(0)
    ncol = nrow
    nrow_fig = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_fig, ncol, figsize=(ncol * 1.4, nrow_fig * 1.4))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(tensor_imgs[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)

    test_imgs_list, test_labels_list = [], []
    for x, y in test_loader:
        test_imgs_list.append(x)
        test_labels_list.append(y)
    test_imgs = torch.cat(test_imgs_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)
    real_fid_subset = test_imgs[:500]

    epochs_std = 25
    epochs_ddpm = 20
    loss_histories = {}
    summary_rows = []

    # 1. VANILLA AUTOENCODER
    print("[1/6] Training Vanilla AE...")
    vae_std = VanillaAE(64).to(device)
    opt = torch.optim.Adam(vae_std.parameters(), lr=1e-3)
    losses, epoch_times = [], []
    for ep in range(epochs_std):
        t0 = time.perf_counter()
        vae_std.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, _ = vae_std(x)
            loss = nn.functional.mse_loss(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        epoch_times.append(time.perf_counter() - t0)
        losses.append(ep_loss / len(train_loader))
    loss_histories["Vanilla AE"] = losses

    vae_std.eval()
    with torch.no_grad():
        recon_van = torch.cat([vae_std(x.to(device))[0].cpu() for x, _ in test_loader], dim=0)
    m_van = compute_reconstruction_metrics(test_imgs, recon_van)
    lat_van = measure_latency_ms(lambda: vae_std.decode(torch.randn(64, 64, device=device)), device)
    summary_rows.append([
        "Vanilla Autoencoder", "64", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_van:.2f}",
        f"{m_van['MSE']:.4f}", f"{m_van['PSNR']:.2f}", f"{m_van['SSIM']:.4f}", "N/A (Latent Voids)", "Stabil / Konvergen"
    ])
    print("[1/6] Selesai Vanilla AE.")

    # 2. SPARSE AUTOENCODER
    print("[2/6] Training Sparse AE...")
    sae = SparseAE(256, l1_lambda=1e-4).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    losses, epoch_times = [], []
    for ep in range(epochs_std):
        t0 = time.perf_counter()
        sae.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, h = sae(x)
            loss, _ = sae.loss_function(x, recon, h)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        epoch_times.append(time.perf_counter() - t0)
        losses.append(ep_loss / len(train_loader))
    loss_histories["Sparse AE"] = losses

    sae.eval()
    with torch.no_grad():
        recon_sparse = torch.cat([sae(x.to(device))[0].cpu() for x, _ in test_loader], dim=0)
    m_sparse = compute_reconstruction_metrics(test_imgs, recon_sparse)
    lat_sparse = measure_latency_ms(lambda: sae.decode(torch.randn(64, 256, device=device)), device)
    summary_rows.append([
        "Sparse Autoencoder", "256 (Sparse)", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_sparse:.2f}",
        f"{m_sparse['MSE']:.4f}", f"{m_sparse['PSNR']:.2f}", f"{m_sparse['SSIM']:.4f}", "N/A (Latent Voids)", "Stabil / Konvergen"
    ])
    print("[2/6] Selesai Sparse AE.")

    # 3. VARIATIONAL AUTOENCODER (VAE)
    print("[3/6] Training VAE & menghitung FID...")
    vae = VAE(64).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    losses, epoch_times = [], []
    for ep in range(epochs_std):
        t0 = time.perf_counter()
        vae.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, _, _ = vae.loss_function(x, recon, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        epoch_times.append(time.perf_counter() - t0)
        losses.append(ep_loss / len(train_loader))
    loss_histories["VAE"] = losses

    vae.eval()
    with torch.no_grad():
        recon_vae = torch.cat([vae(x.to(device))[0].cpu() for x, _ in test_loader], dim=0)
        fake_vae = torch.cat([vae.decode(torch.randn(100, 64, device=device)).cpu() for _ in range(5)], dim=0)
    fid_vae = calculate_fid(real_fid_subset, fake_vae[:500], device)
    m_vae = compute_reconstruction_metrics(test_imgs, recon_vae)
    lat_vae = measure_latency_ms(lambda: vae.decode(torch.randn(64, 64, device=device)), device)
    summary_rows.append([
        "Variational Autoencoder (VAE)", "64", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_vae:.2f}",
        f"{m_vae['MSE']:.4f}", f"{m_vae['PSNR']:.2f}", f"{m_vae['SSIM']:.4f}", f"{fid_vae:.2f}", "Stabil / Konvergen"
    ])
    print(f"[3/6] Selesai VAE (FID={fid_vae:.2f}).")

    # 4. VANILLA GAN
    print("[4/6] Training Vanilla GAN & menghitung FID...")
    vG = VanillaGenerator(100).to(device)
    vD = VanillaDiscriminator().to(device)
    optG = torch.optim.Adam(vG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(vD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce_logits = nn.BCEWithLogitsLoss()
    g_losses, d_losses, epoch_times = [], [], []
    for ep in range(epochs_std):
        t0 = time.perf_counter()
        vG.train(); vD.train()
        gl, dl = 0.0, 0.0
        for x, _ in train_loader:
            real = x.to(device) * 2.0 - 1.0
            b = real.size(0)
            z = torch.randn(b, 100, device=device)
            fake = vG(z).detach()
            lossD = bce_logits(vD(real), torch.ones(b, 1, device=device)) + \
                    bce_logits(vD(fake), torch.zeros(b, 1, device=device))
            optD.zero_grad(); lossD.backward(); optD.step()

            z = torch.randn(b, 100, device=device)
            lossG = bce_logits(vD(vG(z)), torch.ones(b, 1, device=device))
            optG.zero_grad(); lossG.backward(); optG.step()
            gl += lossG.item(); dl += lossD.item()
        epoch_times.append(time.perf_counter() - t0)
        g_losses.append(gl / len(train_loader))
        d_losses.append(dl / len(train_loader))
    loss_histories["Vanilla GAN (G)"] = g_losses
    loss_histories["Vanilla GAN (D)"] = d_losses

    vG.eval()
    with torch.no_grad():
        fake_vgan = torch.cat([((vG(torch.randn(100, 100, device=device)) + 1) / 2).cpu() for _ in range(5)], dim=0)
    fid_vgan = calculate_fid(real_fid_subset, fake_vgan[:500], device)
    lat_vgan = measure_latency_ms(lambda: vG(torch.randn(64, 100, device=device)), device)
    summary_rows.append([
        "Vanilla GAN (MLP)", "100", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_vgan:.2f}",
        "N/A", "N/A", "N/A", f"{fid_vgan:.2f}", "Rentan Osilasi / Mode Collapse"
    ])
    print(f"[4/6] Selesai Vanilla GAN (FID={fid_vgan:.2f}).")

    # 5. DCGAN
    print("[5/6] Training DCGAN & menghitung FID...")
    dcG = DCGANGenerator(100).to(device)
    dcD = DCGANDiscriminator().to(device)
    dcG.apply(weights_init_dcgan); dcD.apply(weights_init_dcgan)
    optG = torch.optim.Adam(dcG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(dcD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_losses, d_losses, epoch_times = [], [], []
    for ep in range(epochs_std):
        t0 = time.perf_counter()
        dcG.train(); dcD.train()
        gl, dl = 0.0, 0.0
        for x, _ in train_loader:
            real = x.to(device) * 2.0 - 1.0
            b = real.size(0)
            z = torch.randn(b, 100, device=device)
            fake = dcG(z).detach()
            lossD = bce_logits(dcD(real), torch.ones(b, 1, device=device)) + \
                    bce_logits(dcD(fake), torch.zeros(b, 1, device=device))
            optD.zero_grad(); lossD.backward(); optD.step()

            z = torch.randn(b, 100, device=device)
            lossG = bce_logits(dcD(dcG(z)), torch.ones(b, 1, device=device))
            optG.zero_grad(); lossG.backward(); optG.step()
            gl += lossG.item(); dl += lossD.item()
        epoch_times.append(time.perf_counter() - t0)
        g_losses.append(gl / len(train_loader))
        d_losses.append(dl / len(train_loader))
    loss_histories["DCGAN (G)"] = g_losses
    loss_histories["DCGAN (D)"] = d_losses

    dcG.eval()
    with torch.no_grad():
        fake_dcgan = torch.cat([((dcG(torch.randn(100, 100, device=device)) + 1) / 2).cpu() for _ in range(5)], dim=0)
    fid_dcgan = calculate_fid(real_fid_subset, fake_dcgan[:500], device)
    lat_dcgan = measure_latency_ms(lambda: dcG(torch.randn(64, 100, device=device)), device)
    summary_rows.append([
        "DCGAN", "100", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_dcgan:.2f}",
        "N/A", "N/A", "N/A", f"{fid_dcgan:.2f}", "Cukup Stabil (Sensitif Arsitektur)"
    ])
    print(f"[5/6] Selesai DCGAN (FID={fid_dcgan:.2f}).")

    # 6. DDPM
    print("[6/6] Training DDPM, sampling & menghitung FID...")
    ddpm = DDPM(timesteps=300, device=device)
    opt = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    losses, epoch_times = [], []
    for ep in range(epochs_ddpm):
        t0 = time.perf_counter()
        ddpm.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x0 = x.to(device) * 2.0 - 1.0
            loss = ddpm.p_losses(x0)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        epoch_times.append(time.perf_counter() - t0)
        losses.append(ep_loss / len(train_loader))
    loss_histories["DDPM"] = losses

    ddpm.eval()
    fake_ddpm = torch.cat([ddpm.sample(100).cpu() for _ in range(5)], dim=0)
    fid_ddpm = calculate_fid(real_fid_subset, fake_ddpm[:500], device)
    lat_ddpm = measure_latency_ms(lambda: ddpm.sample(64), device, repeats=3)
    summary_rows.append([
        "DDPM", "28x28", f"{sum(epoch_times)/len(epoch_times):.2f}", f"{lat_ddpm:.2f}",
        "N/A", "N/A", "N/A", f"{fid_ddpm:.2f}", "Sangat Stabil"
    ])
    print(f"[6/6] Selesai DDPM (FID={fid_ddpm:.2f}).")

    cols = [
        "Model", "Dim Laten", "Train Time/Epoch (s)", "Latency/Batch 64 (ms)",
        "Test MSE", "Test PSNR (dB)", "Test SSIM", "FID Score", "Kestabilan"
    ]
    df = pd.DataFrame(summary_rows, columns=cols)
    df.to_csv("results/metrics_summary.csv", index=False)
    print("\n=== TABEL KOMPARASI METRIK ===")
    print(df.to_string(index=False))

    # FIGUR 1: Grid Rekonstruksi Side-by-Side (8 sampel)
    fig1_grid = torch.cat([test_imgs[:8], recon_van[:8], recon_sparse[:8], recon_vae[:8]], dim=0)
    save_grid(
        fig1_grid, "results/figures/Figur1_Rekonstruksi.png", nrow=8,
        title="Figur 1: Ground Truth (B1), Vanilla AE (B2), Sparse AE (B3), VAE (B4)"
    )

    # FIGUR 2: Random Generative Sampling 4x4 per model
    save_grid(fake_vae[:16], "results/figures/Figur2A_VAE.png", nrow=4, title="Panel A: VAE Random Sampling")
    save_grid(fake_vgan[:16], "results/figures/Figur2B_VanillaGAN.png", nrow=4, title="Panel B: Vanilla GAN Sampling")
    save_grid(fake_dcgan[:16], "results/figures/Figur2C_DCGAN.png", nrow=4, title="Panel C: DCGAN Sampling")
    save_grid(fake_ddpm[:16], "results/figures/Figur2D_DDPM.png", nrow=4, title="Panel D: DDPM Sampling")

    # FIGUR 3: Bukti Empiris Batasan MLP Classifier
    mlp_data = torch.load("results/checkpoints/mlp_proof.pt")
    fig3_grid = torch.cat([
        test_imgs[:8],
        mlp_data["latent_proj"][:8],
        mlp_data["recon_samples"][:8],
        recon_van[:8],
        mlp_data["rand_samples"][:8],
    ], dim=0)
    save_grid(
        fig3_grid, "results/figures/Figur3_MLP_Proof.png", nrow=8,
        title="Figur 3: Asli (B1), Transpose W^T (B2), Frozen MLP Dec (B3), Vanilla AE (B4), Prior Rand MLP (B5)"
    )

    # FIGUR 4: Linear Latent Interpolation (alpha = 0.0 .. 1.0)
    idx_a = (test_labels == 0).nonzero(as_tuple=True)[0][0]
    idx_b = (test_labels == 9).nonzero(as_tuple=True)[0][0]
    img_a = test_imgs[idx_a:idx_a+1].to(device)
    img_b = test_imgs[idx_b:idx_b+1].to(device)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    with torch.no_grad():
        interp_rows = []
        for mod, encode_fn in [
            (vae_std, lambda x: vae_std.encode(x)),
            (sae, lambda x: sae.encode(x)),
            (vae, lambda x: vae.encode(x)[0]),
        ]:
            za, zb = encode_fn(img_a), encode_fn(img_b)
            row = torch.cat([mod.decode((1 - a) * za + a * zb).cpu() for a in alphas], dim=0)
            interp_rows.append(row)

        za_g, zb_g = torch.randn(1, 100, device=device), torch.randn(1, 100, device=device)
        interp_rows.append(torch.cat([((vG((1 - a) * za_g + a * zb_g) + 1) / 2).cpu() for a in alphas], dim=0))
        interp_rows.append(torch.cat([((dcG((1 - a) * za_g + a * zb_g) + 1) / 2).cpu() for a in alphas], dim=0))

    save_grid(
        torch.cat(interp_rows, dim=0), "results/figures/Figur4_Interpolasi.png", nrow=6,
        title="Figur 4: Interpolasi Laten (B1: VanAE, B2: SparseAE, B3: VAE, B4: VanGAN, B5: DCGAN)"
    )

    # FIGUR 5: Training Loss Curves & DDPM Denoising Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, v in loss_histories.items():
        axes[0].plot(range(1, len(v) + 1), v, label=k)
    axes[0].set_title("Kurva Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    _, traj = ddpm.sample(batch_size=1, return_trajectory=True)
    traj_strip = torch.cat([t[0, 0] for t in traj], dim=1).numpy()
    axes[1].imshow(traj_strip, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("DDPM Reverse Denoising Trajectory (x_T -> x_0)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("results/figures/Figur5_Loss_dan_DDPM_Trajectory.png", dpi=300)
    plt.close(fig)
    print("Semua evaluasi dan 5 figur berhasil disimpan di folder results/.")

if __name__ == "__main__":
    main()
    