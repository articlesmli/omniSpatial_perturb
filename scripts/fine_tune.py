import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.transformer import SingleCellTransformer
from models.spatial_gnn import SpatialCrossAttentionFusion
from models.perturbation import PerturbationDiffusionHead
from data.loader import PerturbationDataset


def build_knn_edge_index(coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Builds a k-nearest-neighbor spatial graph from [Batch, 2] cell coordinates."""
    num_nodes = coords.size(0)
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    k = min(k, num_nodes - 1)
    dist = torch.cdist(coords, coords)
    _, knn_idx = torch.topk(dist, k=k + 1, largest=False)
    knn_idx = knn_idx[:, 1:]  # Exclude self-loops
    src = torch.arange(num_nodes, device=coords.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


# 1. Initialize device and models (matching pretraining d_model=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SingleCellTransformer(num_genes=2000, d_model=256)
state_dict = torch.load("omnispatial_transformer_pretrained.pth", map_location=device)
backbone.load_state_dict(state_dict)
backbone.to(device)

spatial_fusion = SpatialCrossAttentionFusion(d_model=256).to(device)
diffusion_head = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100).to(device)

# 2. Load dataset and dataloader
dataset = PerturbationDataset("perturbation_data.h5ad", max_genes=2000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define Noise Schedule (Linear Beta Schedule for 100 diffusion steps)
num_timesteps = 100
betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 4. Define Loss Function and Joint Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    list(backbone.parameters())
    + list(spatial_fusion.parameters())
    + list(diffusion_head.parameters()),
    lr=1e-4,
)

# Helper hook to extract 256-dim latent cell embeddings before final scalar projection
latent_cache = {}
def get_latent_hook(module, inp, out):
    hidden = inp[0]
    latent_cache["hidden"] = hidden.mean(dim=1) if hidden.dim() == 3 else hidden

last_linear = [m for m in backbone.modules() if isinstance(m, nn.Linear)][-1]
hook_handle = last_linear.register_forward_hook(get_latent_hook)

# 5. Multimodal Fine-Tuning Loop
num_epochs = 50
backbone.train()
spatial_fusion.train()
diffusion_head.train()
print(f"Starting multimodal spatial diffusion fine-tuning for {num_epochs} epochs...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for gene_ids, control_expr, target_expr, perturb_ids, ctrl_coords, ctrl_morph in dataloader:
        gene_ids = gene_ids.to(device)
        control_expr = control_expr.to(device)
        target_expr = target_expr.to(device)
        perturb_ids = perturb_ids.to(device)
        ctrl_coords = ctrl_coords.to(device)
        ctrl_morph = ctrl_morph.to(device)
        batch_size = gene_ids.size(0)

        # Extract transcriptomic embeddings [Batch, 256]
        _ = backbone(gene_ids, control_expr)
        z_ctrl_raw = latent_cache["hidden"]

        _ = backbone(gene_ids, target_expr)
        z_target_raw = latent_cache["hidden"]

        # Fuse control embedding with spatial graph neighborhood & morphology features
        edge_index = build_knn_edge_index(ctrl_coords, k=6)
        z_ctrl_fused = spatial_fusion(z_ctrl_raw, edge_index, ctrl_morph)

        # Standardize latent vectors (mean=0, std=1) to match Gaussian noise schedule
        z_ctrl = F.layer_norm(z_ctrl_fused, normalized_shape=(256,))
        z_target = F.layer_norm(z_target_raw, normalized_shape=(256,))

        # Sample random diffusion timesteps and Gaussian noise
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(z_target)

        # Forward diffusion: add noise to z_target to create z_t
        alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar_t) * z_target + torch.sqrt(1.0 - alpha_bar_t) * noise

        # Predict added noise conditioned on multimodal z_ctrl and perturb_ids
        optimizer.zero_grad()
        pred_noise = diffusion_head(z_t, z_ctrl, t, perturb_ids)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Multimodal Diffusion Loss: {epoch_loss / len(dataloader):.4f}")

# Clean up hook and save joint multimodal checkpoint
hook_handle.remove()
torch.save(
    {
        "backbone": backbone.state_dict(),
        "spatial_fusion": spatial_fusion.state_dict(),
        "diffusion_head": diffusion_head.state_dict(),
    },
    "omnispatial_multimodal_finetuned.pth",
)
print("Fine-tuning complete! Model saved to omnispatial_multimodal_finetuned.pth")
