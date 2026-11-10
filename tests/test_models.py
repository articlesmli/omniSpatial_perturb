import torch
from models.spatial_gnn import SpatialCrossAttentionFusion

def test_model_initialization():
    model = SpatialCrossAttentionFusion()
    assert model is not None