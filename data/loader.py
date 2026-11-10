import scanpy as sc
import torch
from torch.utils.data import Dataset
import numpy as np

class PerturbationDataset(Dataset):
    def __init__(self, h5ad_path, perturbation_key='perturbation', control_label='control', max_genes=2000):
        adata = sc.read_h5ad(h5ad_path)
        
        if 'highly_variable' not in adata.var:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=max_genes, flavor='seurat')
            
        adata = adata[:, adata.var['highly_variable']]
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X[~np.isfinite(X)] = 0.0
        
        self.expression = torch.tensor(X, dtype=torch.float32)
        self.gene_ids = torch.arange(X.shape[1]).repeat(X.shape[0], 1)
        
        if perturbation_key in adata.obs:
            self.perturbations = adata.obs[perturbation_key].values
        else:
            self.perturbations = np.array(['control'] * adata.n_obs)
            
        self.control_mask = (self.perturbations == control_label)
        if self.control_mask.sum() > 0:
            self.control_mean = self.expression[self.control_mask].mean(dim=0, keepdim=True)
        else:
            self.control_mean = self.expression.mean(dim=0, keepdim=True)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, idx):
        target_expr = self.expression[idx].unsqueeze(-1)
        control_expr = self.control_mean.squeeze(0).unsqueeze(-1)
        gene_ids = self.gene_ids[idx]
        return gene_ids, control_expr, target_expr
