# GPU server runbook

Run these commands only after copying or cloning the repository onto the target
RTX PRO 6000 Blackwell server. The local development computer is not treated as
the experiment machine.

## 1. Environment

```bash
cp .env.example .env
# Fill .env without committing it.
./scripts/bootstrap_server.sh
```

The bootstrap script installs PyTorch 2.7.0 from the official CUDA 12.8 wheel
index, installs the pinned project dependencies, and runs `pip check`.

## 2. Models

```bash
./scripts/download_models.sh
```

Each download resolves the configured Hub revision to an immutable commit and
writes a local manifest. Model files remain outside Git.

## 3. Deep hardware check

```bash
source .venv/bin/activate
tide preflight --config configs/overnight.yaml --deep
```

Stop if the GPU name is not `RTX PRO 6000 Blackwell`, BF16 backward fails, the
FLAN-T5 batch fails, free disk is inadequate, or credentials are unavailable.

## 4. Controlled data and annotation

```bash
tide build-data --config configs/overnight.yaml
tide annotate --config configs/overnight.yaml --smoke-sources 10 --dry-run
tide annotate --config configs/overnight.yaml --smoke-sources 10
tide annotate --config configs/overnight.yaml --resume
tide validate-data --config configs/overnight.yaml --freeze
```

Set the current DeepSeek input/output prices in `.env` before paid annotation.
The code refuses to estimate cost without them. The cache is content-addressed,
and every retry counts against request and cost caps.

## 5. Implementation gate and formal run

```bash
tide debug --config configs/overnight.yaml --examples 32 --steps 200
tide run --config configs/overnight.yaml --stage all --resume
```

`run` records `runs/overnight/status.json` and stops with a non-zero exit on the
first failed gate. M3 runs only with `--include-optional`.

For manual stage execution, use `tide --help`. Training checkpoints include the
trainable state, optimizer, scheduler, RNG states, exact training position,
effective configuration, data digest, and selection metric. Resume rejects a
mismatched variant or data digest.

## 6. Publication

Inspect generated synthetic data and reports before publishing. Do not commit
`.env`, API caches, model weights, unreviewed raw external samples, or checkpoint
binaries. Large checkpoints should use a release asset or external artifact
store with checksums.
