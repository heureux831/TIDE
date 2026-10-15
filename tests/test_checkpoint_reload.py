import torch

from tide.models import TaskConditionedMatrixBottleneck


def test_bottleneck_checkpoint_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(123)
    original = TaskConditionedMatrixBottleneck(
        hidden_size=16, slots=4, heads=4, ffn_dim=32, gate_hidden_dim=8
    ).eval()
    states = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.long)
    task = torch.randn(2, 16)
    with torch.no_grad():
        expected, _ = original(states, mask, task)
    path = tmp_path / "matrix.pt"
    torch.save(original.state_dict(), path)
    restored = TaskConditionedMatrixBottleneck(
        hidden_size=16, slots=4, heads=4, ffn_dim=32, gate_hidden_dim=8
    ).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual, _ = restored(states, mask, task)
    assert torch.equal(expected, actual)
