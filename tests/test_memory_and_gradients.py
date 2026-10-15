import pytest

torch = pytest.importorskip("torch")

from tide.models import TaskConditionedMatrixBottleneck  # noqa: E402


def test_matrix_bottleneck_shape_mask_and_gradients() -> None:
    torch.manual_seed(7)
    module = TaskConditionedMatrixBottleneck(
        hidden_size=16, slots=4, heads=4, ffn_dim=32, gate_hidden_dim=8
    )
    states = torch.randn(2, 9, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    task = torch.randn(2, 16)
    memory, attention = module(states, mask, task)
    assert memory.shape == (2, 4, 16)
    assert attention.shape == (2, 4, 4, 9)
    assert torch.allclose(attention[..., 4:][0], torch.zeros_like(attention[..., 4:][0]))
    memory.square().mean().backward()
    assert module.query_slots.grad is not None
    assert torch.isfinite(module.query_slots.grad).all()
    assert module.query_slots.grad.abs().sum() > 0


def test_flan_t5_has_exact_decoder_lora_scope() -> None:
    from transformers import T5Config, T5ForConditionalGeneration

    from tide.models import decoder_lora_targets

    config = T5Config(
        vocab_size=100,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    base = T5ForConditionalGeneration(config)
    targets = decoder_lora_targets(base)
    assert len(targets) == 48
    assert all(name.startswith("decoder.") for name in targets)
    assert all(name.endswith((".q", ".v")) for name in targets)


def test_matrix_forward_backprop_and_generation_have_no_encoder_bypass() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import T5Config, T5ForConditionalGeneration

    from tide.models import (
        FlanT5TaskConditionedMatrixRewriter,
        decoder_lora_targets,
    )

    config = T5Config(
        vocab_size=100,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    base = T5ForConditionalGeneration(config)
    base.requires_grad_(False)
    decoder = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=2,
            lora_alpha=4,
            target_modules=decoder_lora_targets(base),
        ),
    )
    bottleneck = TaskConditionedMatrixBottleneck(
        hidden_size=32,
        slots=4,
        heads=4,
        ffn_dim=64,
        gate_hidden_dim=16,
    )
    model = FlanT5TaskConditionedMatrixRewriter(decoder, bottleneck)
    input_ids = torch.randint(2, 100, (2, 8))
    input_mask = torch.ones_like(input_ids)
    context_mask = input_mask.clone()
    task_ids = torch.randint(2, 100, (2, 4))
    task_mask = torch.ones_like(task_ids)
    labels = torch.randint(2, 100, (2, 5))

    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hook = model.encoder.register_forward_hook(count_call)
    output = model(
        input_ids,
        input_mask,
        context_mask,
        task_ids,
        task_mask,
        labels,
    )
    output.loss.backward()
    assert calls == 2
    assert bottleneck.query_slots.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())

    memory, _ = model.encode_memory(input_ids, input_mask, context_mask, task_ids, task_mask)
    calls_before_generation = calls
    generated = model.generate_from_memory(memory, do_sample=False, max_new_tokens=3)
    assert generated.shape[0] == 2
    assert calls == calls_before_generation
    hook.remove()
