import torch
import torch.nn as nn

class SingleCellTransformer(nn.Module):
    def __init__(self, num_genes: int, d_model: int = 256, n_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.gene_embedding = nn.Embedding(num_genes, d_model, padding_idx=0)
        self.value_encoder = nn.Linear(1, d_model) # Encodes binned/normalized expression values
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1) # Predicts expression value or recovers masked token

    def forward(self, gene_ids: torch.Tensor, expression_vals: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        gene_ids: [Batch, SeqLen]
        expression_vals: [Batch, SeqLen, 1]
        """
        x = self.gene_embedding(gene_ids) + self.value_encoder(expression_vals)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        logits = self.head(x)
        return logits
