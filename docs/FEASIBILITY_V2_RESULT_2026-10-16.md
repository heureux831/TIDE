# V2 feasibility result — 2026-10-16

## Outcome

The mandatory v2 pipeline completed successfully as software, and the optional
M3 experiment was also completed. The scientific result is negative: the
64-slot matrix bottleneck did not meet the utility thresholds under the fixed
300-update formal budget. No privacy/utility separation claim is supported.

V2 was declared only after preserving the failed v1 result. Its sole protocol
change was increasing the implementation-gate budget from 200 to 500 updates;
formal training, data, models, losses, seeds, and evaluation remained unchanged.

## Execution status

- Deep GPU preflight: pass.
- 600 annotated task records: schema, semantics, split isolation, and freeze pass.
- V2 debug gate at 500 updates: M1 FactJoint 1.0; M2 FactJoint 1.0.
- Formal M1, M2: 300 updates each, completed from original initialization.
- Optional formal M3: 300 updates from the same independent initialization,
  completed.
- Dev/test Qwen3-8B evaluation, M2/M3 memory diagnostics, and bootstrap report:
  completed.

## Formal model selection

| Variant | Trainable parameters | Best step | Dev FactJoint | Dev text loss |
| --- | ---: | ---: | ---: | ---: |
| M1 decoder LoRA | 1,179,648 | 300 | 1.000 | 0.06172 |
| M2 matrix + decoder LoRA | 7,137,280 | 300 | 0.000 | 0.75008 |
| M3 matrix + auxiliary loss + decoder LoRA | 7,137,280 | 300 | 0.000 | 0.68751 |

M2 text loss decreased at every saved checkpoint, from 2.560 at step 50 to
0.750 at step 300, but exact fact generation never became correct. A read-only
replay of the selected M2 checkpoint produced FactJoint 0.0 on train, dev, and
test; train text loss remained 0.60076. Therefore the immediate failure is
underfitting/optimization of the matrix path, not generalization from seen to
unseen templates.

## Independent Qwen evaluation

### Dev

| Group | TaskAcc | RequiredFactRecall | AnyLeakRate | FactJoint | JointSuccess |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 0.92 | 1.000 | 1.00 | 0.00 | 0.00 |
| EMPTY | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 |
| ORACLE | 0.96 | 1.000 | 0.00 | 1.00 | 0.96 |
| M1 | 0.96 | 1.000 | 0.00 | 1.00 | 0.96 |
| M2 | 0.51 | 0.015 | 0.00 | 0.00 | 0.00 |
| M3 | 0.55 | 0.020 | 0.00 | 0.00 | 0.00 |

### Test

| Group | TaskAcc | RequiredFactRecall | AnyLeakRate | FactJoint | JointSuccess |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 0.92 | 1.000 | 1.00 | 0.00 | 0.00 |
| EMPTY | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 |
| ORACLE | 0.95 | 1.000 | 0.00 | 1.00 | 0.95 |
| M1 | 0.95 | 1.000 | 0.00 | 1.00 | 0.95 |
| M2 | 0.48 | 0.015 | 0.00 | 0.00 | 0.00 |
| M3 | 0.53 | 0.035 | 0.00 | 0.00 | 0.00 |

M2 minus M1 test TaskAcc was -0.47 with a source-grouped 95% bootstrap interval
of [-0.57, -0.37]. Their AnyLeakRate difference was 0. M2's zero measured leak
rate is not a useful privacy result because it also omitted almost every required
fact.

The prespecified evaluator gate was not met: ORACLE reached the required 0.95
TaskAcc, but RAW was 0.92. Consequently the aggregate report correctly marks the
M2 engineering verdict as not evaluable. M2 and M3 nevertheless fail their
internal deterministic fact metrics independently of that evaluator gate.

## Memory-dependence diagnostics

| Variant | Condition | Dev TaskAcc | RequiredFactRecall | Rewrite change vs normal |
| --- | --- | ---: | ---: | ---: |
| M2 | normal | 0.51 | 0.015 | — |
| M2 | shuffled | 0.44 | 0.025 | 0.61 |
| M2 | zero | 0.00 | 0.000 | 1.00 |
| M3 | normal | 0.55 | 0.020 | — |
| M3 | shuffled | 0.48 | 0.020 | 0.85 |
| M3 | zero | 0.00 | 0.000 | 1.00 |

Both models genuinely depend on matrix memory: zeroing it destroys all task
accuracy and shuffling it changes most outputs. The failure is therefore not an
original-text bypass or unused-memory bug.

## Answer: quantity or diversity?

Neither is the first bottleneck demonstrated by this run.

- The same 400 training records are sufficient for M1 to reach 100% dev/test
  FactJoint and 95% test TaskAcc.
- M2 has 0% FactJoint even when replayed on its own training set, so it has not
  reached the stage where unseen-template diversity is the limiting factor.
- The 32-record debug set becomes perfectly memorizable only after increasing
  its optimization budget from 200 to 500 updates.
- M3's weak auxiliary supervision raises test RequiredFactRecall only from 1.5%
  to 3.5%, which is directionally positive but operationally inadequate.

The next experiment should target optimization and representation diversity in
the matrix module before paying for more annotations. Candidate changes must be
versioned separately: stronger or curriculum-based fact-attention supervision,
an explicit slot-diversity objective, or a substantially larger formal training
budget with a prespecified stopping rule. More data becomes relevant only after
the revised model can fit the current training set.
