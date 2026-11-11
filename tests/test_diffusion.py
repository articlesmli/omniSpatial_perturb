import torch
import torch.nn as nn
from models.perturbation import PerturbationDiffusionHead


def test_diffusion_shape_and_gradients():
    torch.manual_seed(42)
    model = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100)

    batch_size = 16
    z_t = torch.randn(batch_size, 256)
    z_ctrl = torch.randn(batch_size, 256)
    t = torch.randint(0, 100, (batch_size,))
    perturb_ids = torch.randint(0, 100, (batch_size,))

    pred_noise = model(z_t, z_ctrl, t, perturb_ids)
    assert pred_noise.shape == (batch_size, 256), f"Expected (16, 256), got {pred_noise.shape}"

    loss = pred_noise.mean()
    loss.backward()
    assert model.net[0].weight.grad is not None, "Gradients failed to flow to the first layer"


def test_diffusion_overfit_single_batch():
    torch.manual_seed(42)
    model = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_size = 16
    z_t = torch.randn(batch_size, 256)
    z_ctrl = torch.randn(batch_size, 256)
    t = torch.randint(0, 100, (batch_size,))
    perturb_ids = torch.randint(0, 100, (batch_size,))
    true_noise = torch.randn(batch_size, 256)

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(z_t, z_ctrl, t, perturb_ids), true_noise)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-4, f"Model failed to overfit single batch; final loss: {loss.item()}"


def test_diffusion_personalization_sensitivity():
    torch.manual_seed(42)
    model = PerturbationDiffusionHead(latent_dim=256, num_perturbations=100)
    model.eval()

    with torch.no_grad():
        z_ctrl_a = torch.randn(1, 256)
        z_ctrl_b = torch.randn(1, 256)
        z_t_same = torch.randn(1, 256)
        t_same = torch.tensor([50])
        drug_same = torch.tensor([10])

        out_a = model(z_t_same, z_ctrl_a, t_same, drug_same)
        out_b = model(z_t_same, z_ctrl_b, t_same, drug_same)
        diff = torch.abs(out_a - out_b).mean().item()

    assert diff > 0.0, "Model ignored untreated control cell (z_ctrl) input"


if __name__ == "__main__":
    test_diffusion_shape_and_gradients()
    test_diffusion_overfit_single_batch()
    test_diffusion_personalization_sensitivity()
    print("All diffusion unit tests passed successfully!")
