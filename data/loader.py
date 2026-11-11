import scanpy as sc
import torch
from torch.utils.data import Dataset
import numpy as np


class PerturbationDataset(Dataset):
    def __init__(
        self,
        h5ad_path: str,
        perturbation_key: str = "perturbation",
        control_label: str = "control",
        max_genes: int = 2000,
        morph_dim: int = 256,
    ):
        adata = sc.read_h5ad(h5ad_path)

        if "highly_variable" not in adata.var:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=max_genes, flavor="seurat")

        adata = adata[:, adata.var["highly_variable"]][:, :max_genes]
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X[~np.isfinite(X)] = 0.0

        self.expression = torch.tensor(X, dtype=torch.float32)
        self.gene_ids = torch.arange(X.shape[1]).repeat(X.shape[0], 1)

        # Map unique perturbation names (drugs / CRISPR edits) to integer IDs
        if perturbation_key in adata.obs:
            self.perturbations = adata.obs[perturbation_key].astype(str).values
        else:
            self.perturbations = np.array([control_label] * adata.n_obs)

        unique_perts = sorted(list(set(self.perturbations)))
        self.pert_to_idx = {name: i for i, name in enumerate(unique_perts)}
        self.idx_to_pert = {i: name for name, i in self.pert_to_idx.items()}
        self.perturb_ids = torch.tensor(
            [self.pert_to_idx[p] for p in self.perturbations], dtype=torch.long
        )

        # Store indices of untreated control cells for single-cell pairing
        self.control_indices = np.where(self.perturbations == control_label)[0]
        if len(self.control_indices) == 0:
            self.control_indices = np.arange(adata.n_obs)

        # Extract 2D spatial coordinates (defaults to zeros if missing)
        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
            self.spatial_coords = torch.tensor(coords[:, :2], dtype=torch.float32)
        else:
            self.spatial_coords = torch.zeros((adata.n_obs, 2), dtype=torch.float32)

        # Extract precomputed cell morphology features (defaults to zeros if missing)
        if "morphology" in adata.obsm:
            morph = np.asarray(adata.obsm["morphology"], dtype=np.float32)
            if morph.shape[1] >= morph_dim:
                morph = morph[:, :morph_dim]
            else:
                pad = np.zeros((adata.n_obs, morph_dim - morph.shape[1]), dtype=np.float32)
                morph = np.concatenate([morph, pad], axis=1)
            self.morphology = torch.tensor(morph, dtype=torch.float32)
        else:
            self.morphology = torch.zeros((adata.n_obs, morph_dim), dtype=torch.float32)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, idx):
        target_expr = self.expression[idx].unsqueeze(-1)

        # Sample a real individual untreated control cell
        ctrl_idx = np.random.choice(self.control_indices)
        control_expr = self.expression[ctrl_idx].unsqueeze(-1)

        gene_ids = self.gene_ids[idx]
        perturb_id = self.perturb_ids[idx]
        ctrl_coord = self.spatial_coords[ctrl_idx]
        ctrl_morph = self.morphology[ctrl_idx]

        return gene_ids, control_expr, target_expr, perturb_id, ctrl_coord, ctrl_morph
