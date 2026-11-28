"""Fused (Hamilton-block real matrix) quaternion layers equal the reference implementations."""
import torch

from src.models.quaternion_lstm import QuaternionLSTM
from src.models.quaternion_ops import QuaternionLinear
from src.models.quaternion_transformer import QLinear


def test_real_matrix_matches_quaternion_linear():
    torch.manual_seed(0)
    q = QuaternionLinear(3, 5).double()
    x = torch.randn(7, 3, 4, dtype=torch.float64)
    fused = x.reshape(7, -1) @ q.real_matrix().T + q.bias.reshape(-1)
    assert torch.allclose(fused, q(x).reshape(7, -1), atol=1e-12)


def test_qlinear_flat_matches_quaternion_linear():
    torch.manual_seed(1)
    layer = QLinear(2, 6).double()
    x = torch.randn(4, 9, 8, dtype=torch.float64)
    ref = layer.q(x.reshape(-1, 2, 4)).reshape(4, 9, 24)
    assert torch.allclose(layer(x), ref, atol=1e-12)


def test_fused_qlstm_matches_loop_with_gradients():
    torch.manual_seed(2)
    m = QuaternionLSTM(4, 6, num_layers=2).double()
    x = torch.randn(5, 11, 4, 4, dtype=torch.float64, requires_grad=True)
    a, _ = m._forward_loop(x)
    b, _ = m._forward_fused(x)
    assert torch.allclose(a, b, atol=1e-12)
    ga = torch.autograd.grad(a.sum(), list(m.parameters()) + [x])
    gb = torch.autograd.grad(b.sum(), list(m.parameters()) + [x])
    assert all(torch.allclose(u, v, atol=1e-10) for u, v in zip(ga, gb))


def test_nonlinear_fusion_variants_differ_from_linear():
    from src.models.hierarchical_qlstm import _FUSION_REGISTRY
    torch.manual_seed(0)
    z = [torch.randn(3, 5) for _ in range(4)]
    for name in ("meta_quaternion", "phm", "dense"):
        lin = _FUSION_REGISTRY[name](5, 4)
        nl = _FUSION_REGISTRY[name + "_act"](5, 4)
        nl.load_state_dict(lin.state_dict())
        assert not torch.allclose(lin(z), nl(z))
