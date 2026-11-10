import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
from models.transformer import SingleCellTransformer
from data.loader import PerturbationDataset

# 1. Load Model and Fine-Tuned Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleCellTransformer(num_genes=2000, d_model=256)
state_dict = torch.load("omnispatial_transformer_finetuned.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 2. Setup Dataset and Validation Loader
dataset = PerturbationDataset("pbmc3k.h5ad", max_genes=2000)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. Evaluation Loop & Metric Accumulation
criterion = nn.MSELoss()
val_loss = 0.0
preds_list = []
targets_list = []

print("Running evaluation and computing correlations...")
with torch.no_grad():
    for gene_ids, control_expr, target_expr in val_loader:
        gene_ids = gene_ids.to(device)
        control_expr = control_expr.to(device)
        target_expr = target_expr.to(device)

        pred_expr = model(gene_ids, control_expr)
        loss = criterion(pred_expr, target_expr)
        val_loss += loss.item()

        preds_list.append(pred_expr.squeeze(-1).cpu().numpy())
        targets_list.append(target_expr.squeeze(-1).cpu().numpy())

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
print(f"Validation MSE Loss: {avg_val_loss:.4f}")
print(f"Mean Cell-wise Pearson Correlation: {np.mean(pearson_corrs):.4f}")
print(f"Mean Cell-wise Spearman Correlation: {np.mean(spearman_corrs):.4f}")
