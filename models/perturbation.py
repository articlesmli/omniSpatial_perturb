import torch
import torch.nn as nn

""" This code is as an AI art restorer that cleans up a blurry,
static-filled photo of a cell to reveal what that cell looks like after taking a drug.
In AI, diffusion models work by taking real data, gradually adding random static (noise)
to it until it is unrecognisable, and then training a neural network to remove that 
static step-by-step. This code is the "brain" that figures out which static to remove."""

class PerturbationDiffusionHead(nn.Module):
    def __init__(self, latent_dim: int = 256, num_perturbations: int = 100):
        super().__init__()
        self.perturbation_embedding = nn.Embedding(num_perturbations, latent_dim)
        
        # Input size changes from (latent_dim * 2 + 1) to (latent_dim * 3 + 1)
        # Because we now glue 3 vectors of size 256 together (z_t, z_ctrl, p_emb) + 1 timer
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 512),  # <-- NEW: * 3 instead of * 2
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(
        self, 
        z_t: torch.Tensor, 
        z_ctrl: torch.Tensor,      #  Added the untreated starting cell input
        t: torch.Tensor, 
        perturb_ids: torch.Tensor
    ):
        """
        z_t: Noisy target cell embedding at diffusion step t [Batch, latent_dim]
        z_ctrl: Clean untreated starting cell embedding [Batch, latent_dim]
        t: Diffusion time step [Batch, 1]
        perturb_ids: Identifier for the CRISPR knockout or compound [Batch]
        """
        p_emb = self.perturbation_embedding(perturb_ids)
        t_enc = t.float().unsqueeze(-1) if t.dim() == 1 else t
        
        # Glue all four clues together: noisy target, healthy start, drug, and timer
        cat_in = torch.cat([z_t, z_ctrl, p_emb, t_enc], dim=-1)  # <-- NEW: Added z_ctrl
        return self.net(cat_in)
