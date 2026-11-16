import pytest
import torch

from heatpde.models.baselines import CNN2d
from heatpde.models.fno import FNO2d

MODELS = {
    "fno": lambda in_c: FNO2d(k_z=4, k_t=8, width=8, in_c=in_c),
    "cnn": lambda in_c: CNN2d(in_c=in_c, width=8, dilations=(1, 2, 4)),
    "cnn_causal": lambda in_c: CNN2d(in_c=in_c, width=8, dilations=(1, 2, 4), causal=True),
}


@pytest.mark.parametrize("name", MODELS)
def test_shapes(name):
    model = MODELS[name](5)
    assert model(torch.randn(2, 5, 8, 64)).shape == (2, 1, 8, 64)


def test_cnn_causal_ignores_future():
    torch.manual_seed(0)
    model = MODELS["cnn_causal"](4)
    x = torch.randn(1, 4, 8, 64)
    x2 = x.clone()
    x2[..., 40:] = torch.randn_like(x2[..., 40:])
    torch.testing.assert_close(model(x)[..., :40], model(x2)[..., :40])


@pytest.mark.parametrize("name", MODELS)
def test_overfits_one_batch(name):
    torch.manual_seed(0)
    model = MODELS[name](4)
    x, y = torch.randn(4, 4, 8, 64), torch.randn(4, 1, 8, 64).cumsum(-1) / 8
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.mse_loss(model(x), y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first
