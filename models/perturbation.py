import torch
import torch.nn as nn

class PerturbationDiffusionHead(nn.Module):
    def __init__(self, latent_dim: int = 256, num_perturbations: int = 100):
        super().__init__()
        self.perturbation_embedding = nn.Embedding(num_perturbations, latent_dim)
        
        # Simple denoiser network predicting the noise vector added to cell states
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, perturb_ids: torch.Tensor):
        """
        z_t: Latent cell embedding at diffusion time step t [Batch, latent_dim]
        t: Diffusion time step [Batch, 1]
        perturb_ids: Identifier for the CRISPR knockout or compound [Batch]
        """
        p_emb = self.perturbation_embedding(perturb_ids)
        t_enc = t.float().unsqueeze(-1) if t.dim() == 1 else t
        
        # Concatenate noisy state, condition, and time step
        cat_in = torch.cat([z_t, p_emb, t_enc], dim=-1)
        return self.net(cat_in)
