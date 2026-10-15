# Feasibility run — 2026-10-16

## Status

**Partial, stopped at the mandatory implementation gate.** The prescribed
32-example, 200-update gate passed for M1 but failed for M2, so formal training
and test-set evaluation were not started. This is the required fail-closed
behavior in `EXPERIMENT.md`.

## Environment and inputs

- Server: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), driver 580.95.05.
- Base environment: Python 3.12.3, PyTorch 2.11.0+cu130; CUDA and BF16
  forward/backward passed.
- Rewriter: `google/flan-t5-base`; evaluator: `Qwen/Qwen3-8B`.
- Both model snapshots were obtained through ModelScope after the official
  Hugging Face endpoint timed out and `hf-mirror.com` sustained only roughly
  0.7–1 MB/s. Server-local manifests contain per-file SHA-256 hashes.
- Controlled data: 300 sources and 600 task records, split by source into
  400/100/100 train/dev/test records. Validation and freezing passed with no
  identity duplicates, split leakage, or template overlap.
- DeepSeek annotation: 300 requests total (10 smoke plus 290 resumed), 324,108
  input tokens, 60,913 output tokens, estimated cost USD 2.2808115. All 600
  expanded records passed schema and semantic validation.

No secret values are included in this report or in model manifests.

## Prespecified debug gate

| Run | Slots | Updates | LR change | Best FactJoint | Best text loss | Gate |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| M1 baseline | n/a | 200 | none | 1.00000 | 0.00032 | pass |
| M2 primary | 64 | 200 | none | 0.03125 | 0.39463 | **fail** |
| M2 slot-count diagnostic | 128 | 200 | none | 0.06250 | 0.56818 | fail |
| M2 optimization diagnostic | 64 | 200 | matrix 0.001, LoRA 0.0005, warmup 5 | 0.00000 | 1.04928 | fail |

The 128-slot diagnostic remained a real compression for the debug examples,
whose contexts contained 133–135 tokens.

M2 learned the requested sentence forms and generally avoided identity fields,
but failed to copy the two required numeric facts. Typical failures repeated one
number in both positions or substituted a number from another example.

## Failure diagnosis

The primary M2 checkpoint showed a slot-collapse signature:

| Statistic | Initial | Step 200 |
| --- | ---: | ---: |
| Mean normalized cross-attention entropy | 0.9999999 | 0.9974898 |
| Unique mean-attention argmax positions per example | 8–15 | 1 for every example |
| Mean off-diagonal memory-slot cosine similarity | 0.83743 | 0.92082 |

Thus the matrix path is connected and trainable, but its slots converge to
highly redundant summaries under the fixed 200-update protocol. Increasing the
slot count or learning rate did not repair the gate.

A separate capacity-only run kept the original 64-slot architecture and
learning rates but used a 600-step scheduler horizon. It reached FactJoint
0.90625 at step 350, 0.96875 at step 450, and 1.0 at step 500. This establishes
that the architecture can memorize the debug set; it does **not** count as a
gate pass because it exceeds the prespecified 200 updates and changes the
scheduler horizon. At the planned formal budget of 300 updates, this diagnostic
was only 0.625 on the debug set.

## Conclusion and next decision

The code path, annotation pipeline, GPU execution, model downloads, and M1
baseline are operational. The fixed M2 training protocol is not feasible within
its stated optimization budget, so no claim about the planned M2 privacy/utility
thresholds is supported and the formal run remains intentionally unexecuted.

Before continuing, revise and version the protocol explicitly. The smallest
evidence-backed option is to increase both the debug and formal optimization
budgets, then restart every formal variant from its original initialization.
An architecture change that directly discourages redundant slots is a separate
experiment and must not be mixed into the current comparison.

## Verification

After adding the multi-provider downloader, the server base environment passed:

```text
ruff: pass
mypy: pass (19 source files)
pytest: 17 passed
model download dry-run: pass
```
