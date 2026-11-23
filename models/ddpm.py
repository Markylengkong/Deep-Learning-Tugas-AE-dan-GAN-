import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_c)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    def __init__(self, base_c=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.down1 = ResBlock(1, base_c, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(base_c, base_c * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ResBlock(base_c * 2, base_c * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c * 2, 2, stride=2)
        self.dec2 = ResBlock(base_c * 4, base_c, time_dim)
        self.up1 = nn.ConvTranspose2d(base_c, base_c, 2, stride=2)
        self.dec1 = ResBlock(base_c * 2, base_c, time_dim)
        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        b = self.bot(self.pool2(d2), t_emb)
        u2 = self.dec2(torch.cat([self.up2(b), d2], dim=1), t_emb)
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1), t_emb)
        return self.out(u1)

class DDPM(nn.Module):
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.model = SimpleUNet().to(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var)

    def p_losses(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, batch_size=64, return_trajectory=False):
        x = torch.randn(batch_size, 1, 28, 28, device=self.device)
        traj = []
        save_steps = set(torch.linspace(self.timesteps - 1, 0, 7).long().tolist())
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)
            beta_t = self.betas[i]
            sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]

            model_mean = sqrt_recip_alpha * (x - (beta_t / sqrt_one_minus_ab) * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(self.posterior_var[i]) * noise
            else:
                x = model_mean

            if return_trajectory and i in save_steps:
                traj.append(((x.clamp(-1, 1) + 1.0) / 2.0).cpu())

        imgs = (x.clamp(-1, 1) + 1.0) / 2.0
        if return_trajectory:
            return imgs, traj
        return imgs