# TIDE

TIDE is a feasibility implementation of task-conditioned privacy rewriting. It
uses a frozen `google/flan-t5-base` encoder, decoder-only LoRA, and an explicit
task-conditioned matrix bottleneck. A fixed `Qwen/Qwen3-8B` model evaluates
whether rewritten context still supports the public task.

This repository implements an experiment; it does not claim that the method has
already succeeded, provides differential privacy, or has been validated on real
users, Chinese text, or medical data.

## Experiment names

- `M1_FLAN_T5_LORA`: frozen FLAN-T5 encoder plus decoder q/v LoRA.
- `M2_FLAN_T5_MATRIX`: the same decoder LoRA, reading only 64 task-conditioned
  matrix slots.
- `M3_FLAN_T5_MATRIX_AUX`: optional independent run with attention supervision.

The Python classes use equally explicit names: `FlanT5LoRARewriter` and
`FlanT5TaskConditionedMatrixRewriter`.

## Local setup

The supported Python range is 3.10-3.12. The current development Mac may be used
for deterministic data and unit tests, but GPU acceptance, training, and Qwen
evaluation must run on the target NVIDIA server.

```bash
cp .env.example .env
# Fill .env locally. Never commit it.
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Record the environment and generate the controlled dataset:

```bash
tide preflight --config configs/overnight.yaml
tide build-data --config configs/overnight.yaml
tide validate-data --config configs/overnight.yaml --freeze
```

The primary feasibility dataset is generated deterministically: 300 fictional
profiles, two tasks per profile, grouped into 400/100/100 task records. It does
not download or train on WildChat or HealthBench.

## Download scripts

Preview model downloads without network writes:

```bash
tide-download-models --model all --dry-run
```

Download immutable snapshots after filling `.env`:

```bash
tide-download-models --model flan-t5
tide-download-models --model qwen
```

Each model directory gets `tide_download_manifest.json` containing the resolved
Hub commit. Downloaded weights live under ignored `models/`.

For later extensions, small auditable source samples can be downloaded separately:

```bash
tide-download-data --source healthbench --limit 100
tide-download-data --source wildchat --limit 100
```

These external samples are ignored by Git and their manifests explicitly say
they are not part of the controlled overnight experiment. HealthBench is fetched
from OpenAI's public simple-evals URL; WildChat streams from `allenai/WildChat`.
Review source terms before redistribution.

## Reference annotation

Fill the DeepSeek key and current input/output prices in `.env`. The price fields
are mandatory so a stale hard-coded price cannot silently bypass the budget.

```bash
tide annotate --config configs/overnight.yaml --smoke-sources 10 --dry-run
tide annotate --config configs/overnight.yaml --smoke-sources 10
tide annotate --config configs/overnight.yaml --resume
```

Requests contain both tasks for one synthetic profile. Responses are cached by
content hash, schema-validated, checked against deterministic answers, and
bounded by both request count and estimated cost. Secret values are never put in
manifests or logs.

On the GPU server, after the required smoke gates pass:

```bash
tide train --config configs/overnight.yaml --variant M1_FLAN_T5_LORA
tide train --config configs/overnight.yaml --variant M2_FLAN_T5_MATRIX
tide evaluate --config configs/overnight.yaml --split dev --groups RAW EMPTY ORACLE M1_FLAN_T5_LORA M2_FLAN_T5_MATRIX
tide evaluate --config configs/overnight.yaml --split test --groups RAW EMPTY ORACLE M1_FLAN_T5_LORA M2_FLAN_T5_MATRIX
tide report --config configs/overnight.yaml
```

## Development checks

```bash
ruff check .
mypy src/tide
pytest
```

The deterministic fictional source ledger is committed for auditability.
Downloaded samples, annotation caches, model weights, runs, and checkpoints are
excluded from normal Git history. Publish reviewed annotated data and reports
deliberately; use a release or external artifact store for large checkpoints.

See [the experiment contract](docs/EXPERIMENT.md) for the fixed scientific scope
and [the server runbook](docs/SERVER_RUNBOOK.md) for the exact handoff sequence.
