import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.transformer import SingleCellTransformer
from data.loader import PerturbationDataset

# 1. Initialize device and model (matching pretraining d_model=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleCellTransformer(num_genes=2000, d_model=256)

# 2. Load pretrained weights
state_dict = torch.load("omnispatial_transformer_pretrained.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# 3. Load dataset and dataloader using modular loader
dataset = PerturbationDataset("pbmc3k.h5ad", max_genes=2000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 5. Fine-Tuning Loop
model.train()
print("Starting perturbation fine-tuning...")
for epoch in range(3):
    epoch_loss = 0.0
    for gene_ids, control_expr, target_expr in dataloader:
        gene_ids = gene_ids.to(device)
        control_expr = control_expr.to(device)
        target_expr = target_expr.to(device)

        optimizer.zero_grad()
        pred_expr = model(gene_ids, control_expr)
        loss = criterion(pred_expr, target_expr)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/3] | Fine-Tuning Loss: {epoch_loss / len(dataloader):.4f}")

# 6. Save checkpoint
torch.save(model.state_dict(), "omnispatial_transformer_finetuned.pth")
print("Fine-tuning complete! Model saved to omnispatial_transformer_finetuned.pth")
