import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.transformer import SingleCellTransformer
from models.perturbation import PerturbationDiffusionHead
from data.loader import PerturbationDataset

# 1. Initialize device and models (matching pretraining d_model=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SingleCellTransformer(num_genes=2000, d_model=256)
state_dict = torch.load("omnispatial_transformer_pretrained.pth", map_location=device)
backbone.load_state_dict(state_dict)
backbone.to(device)

# Initialize the updated diffusion head conditioned on z_ctrl
diffusion_head = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100).to(device)

# 2. Load dataset and dataloader
dataset = PerturbationDataset("pbmc3k.h5ad", max_genes=2000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define Noise Schedule (Linear Beta Schedule for 100 diffusion steps)
num_timesteps = 100
betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 4. Define Loss Function and Optimizer (trains both backbone and diffusion head)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    list(backbone.parameters()) + list(diffusion_head.parameters()), 
    lr=1e-4
)

# Helper to extract the 256-dim latent embeddings right before the final scalar head
latent_cache = {}
def get_latent_hook(module, inp, out):
    hidden = inp[0]
    latent_cache["hidden"] = hidden.mean(dim=1) if hidden.dim() == 3 else hidden

last_linear = [m for m in backbone.modules() if isinstance(m, nn.Linear)][-1]
hook_handle = last_linear.register_forward_hook(get_latent_hook)

# 5. Fine-Tuning Loop
num_epochs = 50
backbone.train()
diffusion_head.train()
print(f"Starting personalized diffusion fine-tuning for {num_epochs} epochs...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        # Unpack batch (handles optional perturb_ids if your loader provides them)
        if len(batch) == 4:
            gene_ids, control_expr, target_expr, perturb_ids = batch
        else:
            gene_ids, control_expr, target_expr = batch
            perturb_ids = torch.zeros(gene_ids.size(0), dtype=torch.long)

        gene_ids = gene_ids.to(device)
        control_expr = control_expr.to(device)
        target_expr = target_expr.to(device)
        perturb_ids = perturb_ids.to(device)
        batch_size = gene_ids.size(0)

        # Extract and standardize 256-dim latent embeddings (mean=0, std=1)
        _ = backbone(gene_ids, control_expr)
        z_ctrl = F.layer_norm(latent_cache["hidden"], normalized_shape=(256,))

        _ = backbone(gene_ids, target_expr)
        z_target = F.layer_norm(latent_cache["hidden"], normalized_shape=(256,))

        # Sample random diffusion timesteps and Gaussian noise
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(z_target)

        # Forward diffusion: add noise to z_target to create z_t
        alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar_t) * z_target + torch.sqrt(1.0 - alpha_bar_t) * noise

        # Predict the added noise conditioned on z_ctrl and perturb_ids
        optimizer.zero_grad()
        pred_noise = diffusion_head(z_t, z_ctrl, t, perturb_ids)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Diffusion Fine-Tuning Loss: {epoch_loss / len(dataloader):.4f}")

# Clean up hook and save both checkpoints
hook_handle.remove()
torch.save({
    "backbone": backbone.state_dict(),
    "diffusion_head": diffusion_head.state_dict()
}, "omnispatial_diffusion_finetuned.pth")
print("Fine-tuning complete! Model saved to omnispatial_diffusion_finetuned.pth")
