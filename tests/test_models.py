import torch
from models.spatial_gnn import SpatialGNN # Adjust to your actual class name

def test_model_initialization():
    model = SpatialGNN()
    assert model is not None
