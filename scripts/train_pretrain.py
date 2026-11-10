import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.transformer import SingleCellTransformer
from data.loader import load_anndata_to_tensors

def mask_gene_tokens(gene_ids: torch.Tensor, expression_vals: torch.Tensor, mask_prob: float = 0.15):
    """
    Applies Masked Gene Modeling (MGM) by masking a percentage of gene tokens.
    """
    batch_size, seq_len = gene_ids.shape
    mask = torch.rand((batch_size, seq_len)) < mask_prob
    
    # Copy original values for target calculation
    target_vals = expression_vals.clone()
    
    # Replace masked expression values with zero (or a special token embedding)
    masked_expression_vals = expression_vals.clone()
    masked_expression_vals[mask] = 0.0
    
    return gene_ids, masked_expression_vals, target_vals, mask

def train_foundation_model(h5ad_path: str, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
    print("Loading and preprocessing single-cell data...")
    gene_ids, expression_vals = load_anndata_to_tensors(h5ad_path)
    
    dataset = TensorDataset(gene_ids, expression_vals)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_genes = int(gene_ids.max().item()) + 1
    
    model = SingleCellTransformer(num_genes=num_genes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Starting pretraining on {device}...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_g, batch_x in dataloader:
            batch_g, batch_x = batch_g.to(device), batch_x.to(device)
            
            # Apply Masked Gene Modeling
            g_ids, masked_x, targets, mask = mask_gene_tokens(batch_g, batch_x)
            
            optimizer.zero_grad()
            preds = model(g_ids, masked_x)
            
            # Compute loss only on masked positions
            loss = criterion(preds[mask], targets[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss / len(dataloader):.4f}")
        
    # Save pretrained weights
    torch.save(model.state_dict(), "omnispatial_transformer_pretrained.pth")
    print("Pretraining complete! Model saved to omnispatial_transformer_pretrained.pth")

if __name__ == "__main__":
    train_foundation_model("pbmc3k.h5ad")

