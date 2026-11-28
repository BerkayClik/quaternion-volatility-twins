"""Depth suffix (-L<n>) and the temporal CNN twins of the paper model zoo."""
import pytest
import torch
import torch.nn.functional as F

from src.models.quaternion_cnn import QuaternionConv1d
from src.models.quaternion_ops import QuaternionLinear, hamilton_product
from src.paper.models import build, build_matched, n_params, split_depth, target_budget

BUDGET = 20_000
WINDOWS = (30, 42, 48)          # daily, 4-hourly and hourly windows of the paper


def test_quaternion_conv_matches_explicit_hamilton_products():
    torch.manual_seed(0)
    in_q, out_q, k, steps = 3, 2, 3, 7
    conv = QuaternionConv1d(in_q, out_q, k).double()
    with torch.no_grad():
        conv.bias.normal_()
    x = torch.randn(4, steps, in_q, 4, dtype=torch.float64)            # (batch, T, quaternion, component)
    fused = conv(x.reshape(4, steps, -1).transpose(1, 2))             # (batch, 4 * out_q, T)
    fused = fused.transpose(1, 2).reshape(4, steps, out_q, 4)

    padded = F.pad(x, (0, 0, 0, 0, k // 2, k // 2))                   # zero quaternions before and after
    ref = torch.zeros(4, steps, out_q, 4, dtype=torch.float64)
    for t in range(steps):
        for o in range(out_q):
            for i in range(in_q):
                for tap in range(k):
                    w = conv.weight[o, i, tap].expand(4, 4)
                    ref[:, t, o] += hamilton_product(padded[:, t + tap, i], w)   # x (x) w, as QuaternionLinear
            ref[:, t, o] += conv.bias[o]
    assert torch.allclose(fused, ref, atol=1e-12)


def test_quaternion_conv_with_one_tap_is_a_quaternion_linear_layer():
    torch.manual_seed(1)
    conv = QuaternionConv1d(3, 5, kernel_size=1).double()
    lin = QuaternionLinear(3, 5).double()
    with torch.no_grad():
        lin.weight.copy_(conv.weight[:, :, 0])
    x = torch.randn(6, 3, 4, dtype=torch.float64)
    out = conv(x.reshape(6, 12, 1)).reshape(6, 5, 4)
    assert torch.allclose(out, lin(x), atol=1e-12)


@pytest.mark.parametrize("name, hidden, params", [
    # Table "Hidden sizes and trainable parameters at the 20k budget" of paper v1 (PMSE, window 42)
    ("lstm", 62, 19_903), ("qlstm", 29, 19_663), ("transformer", 8, 19_073), ("qtransformer", 8, 20_993),
    ("hl_meta", 27, 20_251), ("hq_meta", 14, 20_077), ("ht_meta", 5, 18_251), ("hqt_meta", 9, 20_611),
])
def test_v1_models_are_unchanged(name, hidden, params):
    net, h = build_matched(name, 4, BUDGET, seq_len=42)
    assert (h, n_params(net)) == (hidden, params)


def test_split_depth():
    assert split_depth("lstm") == ("lstm", None)
    assert split_depth("hq_meta-L2") == ("hq_meta", 2)
    assert split_depth("qcnn-L6") == ("qcnn", 6)
    assert split_depth("lstm-L0") == ("lstm-L0", None)      # depth 0 is not a depth suffix


def test_depth_suffix_sets_the_number_of_layers():
    assert build("lstm-L2", 4, 16).lstm.num_layers == 2
    assert build("qlstm-L3", 4, 8).qlstm.num_layers == 3
    assert all(m.num_layers == 2 for m in build("hl_meta-L2", 4, 8).lstms)
    assert all(m.num_layers == 2 for m in build("hq_meta-L2", 4, 6).qlstms)
    assert len(build("transformer-L3", 4, 8, seq_len=42).encoder.layers) == 3
    assert len(build("qtransformer-L3", 4, 8, seq_len=42).encoder.layers) == 3
    assert len(build("hqt_meta-L2", 4, 8, seq_len=42).encoders[0].layers) == 2
    # the default depth and its explicit suffix are the same model
    for plain, explicit in [("lstm", "lstm-L1"), ("hq_meta", "hq_meta-L1"), ("transformer", "transformer-L2"),
                            ("ht_meta", "ht_meta-L1"), ("qcnn", "qcnn-L4"), ("hc_meta", "hc_meta-L4")]:
        assert n_params(build(plain, 4, 12, seq_len=42)) == n_params(build(explicit, 4, 12, seq_len=42))


TWINS = [                        # (real, quaternion)
    ("cnn", "qcnn"), ("cnn-L2", "qcnn-L2"), ("cnn-L6", "qcnn-L6"),
    ("hc_meta", "hqc_meta"), ("hc_concat", "hqc_concat"), ("hc_gattn", "hqc_gattn"),
    ("hc_meta-L2", "hqc_meta-L2"), ("hc_meta-L6", "hqc_meta-L6"),
    ("lstm-L2", "qlstm-L2"), ("lstm-L3", "qlstm-L3"), ("hl_meta-L2", "hq_meta-L2"), ("hl_meta-L3", "hq_meta-L3"),
]


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("real, quaternion", TWINS)
def test_matched_twins_have_close_parameter_counts_and_train(real, quaternion, window):
    """The quaternion model is sized to the budget and its real twin to the quaternion's count."""
    match = {real: quaternion}
    torch.manual_seed(0)
    x = torch.randn(5, window, 16)
    counts = {}
    for name in (quaternion, real):
        budget = target_budget(name, 4, BUDGET, match, seq_len=window)
        net, _ = build_matched(name, 4, budget, seq_len=window)
        counts[name] = n_params(net)
        out = net(x)
        assert out.shape == (5, 1) and torch.isfinite(out).all()
        out.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert abs(counts[quaternion] - BUDGET) / BUDGET < 0.16, counts      # integer widths: steps of up to 25%
    assert abs(counts[real] - counts[quaternion]) / counts[quaternion] < 0.06, counts


def test_target_budget_without_a_twin_is_the_common_budget():
    assert target_budget("lstm", 4, BUDGET, None, seq_len=42) == BUDGET
    assert target_budget("lstm", 4, BUDGET, {"hc_meta": "hqc_meta"}, seq_len=42) == BUDGET
    assert target_budget("hc_meta", 4, BUDGET, {"hc_meta": "hqc_meta"}, seq_len=42) == 22_097


def test_cnn_twins_handle_price_only_and_two_group_inputs():
    for n_groups, names in [(1, ("cnn", "qcnn")), (2, ("cnn", "qcnn", "hc_concat", "hqc_concat"))]:
        for name in names:
            net, _ = build_matched(name, n_groups, BUDGET, seq_len=42)
            assert net(torch.randn(3, 42, 4 * n_groups)).shape == (3, 1)


def test_calendar_suffix_adds_the_target_bar_calendar_to_the_base_forecast():
    from src.paper.models import WithCalendar
    torch.manual_seed(0)
    net = build("hq_meta-L2+cal", 5, 6, seq_len=42)                 # four source groups + calendar
    assert isinstance(net, WithCalendar)
    assert n_params(net) == n_params(build("hq_meta-L2", 4, 6, seq_len=42)) + n_params(net.calendar)
    x = torch.randn(4, 42, 20)
    out = net(x)
    assert out.shape == (4, 1)
    earlier = x.clone()
    earlier[:, :-1, -4:] += 1.0                                       # calendar of earlier window steps is not used
    assert torch.allclose(net(earlier), out)
    later = x.clone()
    later[:, -1, -4:] += 1.0                                          # the calendar of the target bar is
    assert not torch.allclose(net(later), out)
    sources = x.clone()
    sources[..., :16] = 0.0                                           # the base model sees only the 16 source features
    assert torch.allclose(net(sources) - net.calendar(x[:, -1, -4:]), net.base(torch.zeros(4, 42, 16)))


def test_calendar_models_fit_the_runner_rules_and_budget():
    import importlib.util
    spec = importlib.util.spec_from_file_location("paper_run", "experiments/paper_run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    P4 = ["price", "market", "social", "sentiment"]
    assert runner._fits("hq_meta-L2+cal", P4 + ["calendar"])            # meta fusion: 4 source groups + calendar
    assert not runner._fits("hq_meta+cal", ["price", "market", "social", "calendar"])
    assert runner._fits("lstm+cal", ["price", "calendar"]) and not runner._fits("hl_meta+cal", ["price", "calendar"])
    assert not runner._fits("lstm+cal", P4)                             # the last group must be the calendar
    assert runner._fits("hq_meta-L2", P4) and not runner._fits("hq_meta-L2", P4 + ["calendar"])
    match = {"hl_meta-L2+cal": "hq_meta-L2+cal"}
    q = n_params(build_matched("hq_meta-L2+cal", 5, BUDGET, seq_len=42)[0])
    r = n_params(build_matched("hl_meta-L2+cal", 5, target_budget("hl_meta-L2+cal", 5, BUDGET, match, seq_len=42), seq_len=42)[0])
    assert abs(q - BUDGET) / BUDGET < 0.05 and abs(r - q) / q < 0.03
