# Controlled feasibility experiment

This repository answers a narrow question: can a FLAN-T5 rewriter generate a
useful task-specific context when its decoder can read only a learned matrix
bottleneck, and does that model show a better utility/direct-disclosure tradeoff
than an otherwise matched decoder-LoRA baseline?

## Fixed scope

- 300 deterministic fictional profiles, each paired with two public tasks.
- Source-grouped 200/50/50 train/dev/test split, producing 400/100/100 task rows.
- Sixteen train, four dev, and four test expression templates with no overlap.
- Programmatic YES/NO answers and exact joint-label quotas.
- DeepSeek supplies reference rewrites and a separately checked annotation answer;
  it does not define the final gold answer.
- `M1_FLAN_T5_LORA` and `M2_FLAN_T5_MATRIX` are mandatory.
- `M3_FLAN_T5_MATRIX_AUX` is optional and starts from the same independent initial
  state, never from the selected M2 checkpoint.
- Qwen3-8B performs independent downstream task evaluation and never receives
  source IDs, fact IDs, protected/required labels, gold answers, or original text
  when evaluating a rewrite.

This run does not establish differential privacy, resistance to attribute
inference, performance on real people, or generalization to Chinese or medical
text.

## Model path

`FlanT5LoRARewriter` freezes the FLAN-T5 encoder and original decoder weights,
then trains rank-16 LoRA adapters on exactly the 48 decoder self/cross-attention
q/v projections.

`FlanT5TaskConditionedMatrixRewriter` encodes the formatted task/context and the
task alone with the same frozen encoder. Sixty-four learned queries attend only
to context-token states. A residual FFN and task-conditioned elementwise gate
produce `Z` with shape `[batch, 64, 768]`. The T5 decoder receives only `Z` via
`encoder_outputs`; no original encoder memory is concatenated or re-encoded
during generation.

## Prespecified gates

Before formal training:

1. Deep GPU preflight must verify the exact GPU identity, BF16 forward/backward,
   and a real FLAN-T5 batch.
2. Ten-source DeepSeek annotation smoke must pass strict schema and semantic
   validation.
3. Tiny-model tests must prove decoder-only LoRA selection, matrix gradients,
   encoder freezing, checkpoint equality, padding masking, and absence of a
   generation-time encoder bypass.
4. M1 and M2 must each overfit 32 task examples to at least 90% FactJoint.

Final test thresholds are fixed before results: RAW and ORACLE TaskAcc ≥95%; M2
TaskAcc ≥90% and within five points of RAW; M2 RequiredFactRecall ≥95%,
AnyLeakRate ≤10%, and JointSuccess ≥80%. Meeting those values means the controlled
implementation is worth continuing. It does not by itself show M2 beats M1.

The report separately states whether the code path ran, whether engineering
thresholds were met, and whether an additional M2 benefit over M1 was supported.

## Versioned continuation after the first gate failure

The 2026-10-16 v1 run is preserved as a failed prespecified experiment: M1
reached 100% FactJoint at 200 updates while M2 reached 3.125%. A separately
labelled capacity diagnostic showed that unchanged M2 reached 100% at 500
updates under a 600-step scheduler horizon. It is therefore permissible to run
the post-hoc `configs/overnight_v2.yaml` continuation with a 500-update debug
budget. No other model, data, loss, formal-training, or evaluation setting is
changed. V2 is reported separately and does not retroactively turn v1 into a
pass.
