import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
from models.transformer import SingleCellTransformer
from models.perturbation import PerturbationDiffusionHead
from data.loader import PerturbationDataset

# 1. Initialize Device and Load Joint Checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SingleCellTransformer(num_genes=2000, d_model=256).to(device)
diffusion_head = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100).to(device)

checkpoint = torch.load("omnispatial_diffusion_finetuned.pth", map_location=device)
backbone.load_state_dict(checkpoint["backbone"])
diffusion_head.load_state_dict(checkpoint["diffusion_head"])

backbone.eval()
diffusion_head.eval()

# 2. Setup Dataset and Validation Loader
dataset = PerturbationDataset("pbmc3k.h5ad", max_genes=2000)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. Define Reverse Diffusion Schedule (Matches fine_tune.py)
num_timesteps = 100
betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Helper hook to extract 256-dim latent cell representations
latent_cache = {}
def get_latent_hook(module, inp, out):
    hidden = inp[0]
    latent_cache["hidden"] = hidden.mean(dim=1) if hidden.dim() == 3 else hidden

last_linear = [m for m in backbone.modules() if isinstance(m, nn.Linear)][-1]
hook_handle = last_linear.register_forward_hook(get_latent_hook)

@torch.no_grad()
def sample_perturbed_latent(z_ctrl: torch.Tensor, perturb_ids: torch.Tensor) -> torch.Tensor:
    """Runs reverse DDPM sampling conditioned on z_ctrl and perturb_ids."""
    batch_size = z_ctrl.size(0)
    z_t = torch.randn_like(z_ctrl)

    for step in reversed(range(num_timesteps)):
        t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
        pred_noise = diffusion_head(z_t, z_ctrl, t_batch, perturb_ids)

        alpha_t = alphas[step]
        alpha_bar_t = alphas_cumprod[step]
        beta_t = betas[step]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (z_t - coef2 * pred_noise)

        if step > 0:
            z_t = mean + torch.sqrt(beta_t) * torch.randn_like(z_t)
        else:
            z_t = mean

    return z_t

# 4. Evaluation Loop & Metric Accumulation
criterion = nn.MSELoss()
val_loss = 0.0
preds_list = []
targets_list = []

print("Running reverse diffusion evaluation and computing correlations...")
with torch.no_grad():
    for batch in val_loader:
        if len(batch) == 4:
            gene_ids, control_expr, target_expr, perturb_ids = batch
        else:
            gene_ids, control_expr, target_expr = batch
            perturb_ids = torch.zeros(gene_ids.size(0), dtype=torch.long)

        gene_ids = gene_ids.to(device)
        control_expr = control_expr.to(device)
        target_expr = target_expr.to(device)
        perturb_ids = perturb_ids.to(device)

        # Encode and standardize control and ground-truth target cells (mean=0, std=1)
        _ = backbone(gene_ids, control_expr)
        z_ctrl = F.layer_norm(latent_cache["hidden"], normalized_shape=(256,))

        _ = backbone(gene_ids, target_expr)
        z_target = F.layer_norm(latent_cache["hidden"], normalized_shape=(256,))

        # Generate predicted perturbed latent state from z_ctrl
        z_pred = sample_perturbed_latent(z_ctrl, perturb_ids)

        loss = criterion(z_pred, z_target)
        val_loss += loss.item()

        preds_list.append(z_pred.cpu().numpy())
        targets_list.append(z_target.cpu().numpy())

hook_handle.remove()

preds = np.concatenate(preds_list, axis=0)
targets = np.concatenate(targets_list, axis=0)

pearson_corrs = []
spearman_corrs = []
for i in range(preds.shape[0]):
    p_corr, _ = pearsonr(preds[i], targets[i])
    s_corr, _ = spearmanr(preds[i], targets[i])
    if not np.isnan(p_corr):
        pearson_corrs.append(p_corr)
    if not np.isnan(s_corr):
        spearman_corrs.append(s_corr)

avg_val_loss = val_loss / len(val_loader)
print(f"Validation Latent MSE Loss: {avg_val_loss:.4f}")
print(f"Mean Cell-wise Pearson Correlation: {np.mean(pearson_corrs):.4f}")
print(f"Mean Cell-wise Spearman Correlation: {np.mean(spearman_corrs):.4f}")
