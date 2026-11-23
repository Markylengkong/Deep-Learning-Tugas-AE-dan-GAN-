# Studi Komparatif Model Generatif & Batasan MLP Classifier (Deep Learning D6421)

Repositori proyek eksperimen komparatif untuk mata kuliah **Deep Learning (D6421)** mencakup:
- Pembuktian teoretis dan empiris batasan **MLP Classifier** dalam rekonstruksi & generasi citra.
- Implementasi dan evaluasi terstandarisasi pada dataset **Fashion-MNIST** (split 80/10/10, `seed=42`) untuk 6 model: **Vanilla Autoencoder**, **Sparse Autoencoder**, **Variational Autoencoder (VAE)**, **Vanilla GAN**, **DCGAN**, dan **DDPM**.

## Struktur Direktori
- `src/utils.py`: *Seed*, *dataloader* 80/10/10, metrik rekonstruksi (MSE, PSNR, SSIM), FID, dan pengukuran *latency* per batch 64.
- `models/`: Arsitektur `MLPClassifier`, `FrozenLatentDecoder`, `VanillaAE`, `SparseAE`, `VAE`, `VanillaGenerator/Discriminator`, `DCGANGenerator/Discriminator`, dan `DDPM`.
- `experiments/`: Skrip eksperimen `run_mlp_proof.py` dan `train_and_evaluate.py`.
- `results/`: Tabel metrik (`metrics_summary.csv`), *checkpoint*, dan 5 figur visualisasi utama.

## Cara Menjalankan
1. Aktifkan *virtual environment* dan pasang dependensi:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt