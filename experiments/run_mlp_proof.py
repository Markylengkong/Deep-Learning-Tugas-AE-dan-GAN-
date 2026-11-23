import torch
import torch.nn as nn
from src.utils import set_seed, get_device, get_dataloaders, compute_reconstruction_metrics
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder

def main():
    set_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)

    clf = MLPClassifier(784, 256, 64, 10).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = clf(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    clf.eval()
    correct, total = 0, 0
    all_z, all_x = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, z = clf(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_z.append(z)
            all_x.append(x)

    acc = 100.0 * correct / total
    print(f"[MLP Classifier] Test Accuracy: {acc:.2f}% (Syarat >= 88%)")
    assert acc >= 88.0, "Akurasi belum mencapai 88%, tambah epoch!"

    y_onehot = torch.eye(10, device=device)
    class_proj = clf.top_down_from_class(y_onehot)

    test_x = torch.cat(all_x, dim=0)
    test_z = torch.cat(all_z, dim=0)
    latent_proj = clf.transpose_from_latent(test_z[:16])

    for p in clf.parameters():
        p.requires_grad = False

    decoder = FrozenLatentDecoder(64, 256, 784).to(device)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    for epoch in range(1, 21):
        decoder.train()
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z = clf.encode(x)
            recon = decoder(z)
            loss = mse_loss(recon, x)
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()

    decoder.eval()
    with torch.no_grad():
        recon_test = decoder(test_z)
        metrics = compute_reconstruction_metrics(test_x, recon_test)
        print(f"[Frozen MLP Decoder] Test Metrics: {metrics}")

        mu_z = test_z.mean(dim=0, keepdim=True)
        std_z = test_z.std(dim=0, keepdim=True) + 1e-6
        z_rand = mu_z + std_z * torch.randn(16, 64, device=device)
        rand_samples = decoder(z_rand)

    torch.save({
        "clf_state": clf.state_dict(),
        "decoder_state": decoder.state_dict(),
        "class_proj": class_proj.cpu(),
        "latent_proj": latent_proj.cpu(),
        "recon_samples": recon_test[:16].cpu(),
        "rand_samples": rand_samples.cpu(),
        "metrics": metrics,
        "accuracy": acc,
    }, "results/checkpoints/mlp_proof.pt")

if __name__ == "__main__":
    main()