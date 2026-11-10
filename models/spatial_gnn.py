import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SpatialCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.spatial_conv = GCNConv(d_model, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cell_embeddings: torch.Tensor, edge_index: torch.Tensor, morphology_features: torch.Tensor):
        """
        cell_embeddings: [NumCells, d_model] from the transformer backbone
        edge_index: [2, NumEdges] spatial adjacency representation
        morphology_features: [NumCells, d_model] extracted image features from cell boundaries
        """
        # 1. Update cell representations using local spatial graph neighborhoods
        spatial_context = self.spatial_conv(cell_embeddings, edge_index)
        
        # 2. Fuse transcriptomic spatial context with image/morphology features via Cross-Attention
        # Query: spatial transcriptomics, Key/Value: cell morphology features
        query = spatial_context.unsqueeze(1) # [NumCells, 1, d_model]
        kv = morphology_features.unsqueeze(1) # [NumCells, 1, d_model]
        
        attn_out, _ = self.cross_attention(query=query, key=kv, value=kv)
        fused = self.norm(cell_embeddings + attn_out.squeeze(1))
        
        return fused
