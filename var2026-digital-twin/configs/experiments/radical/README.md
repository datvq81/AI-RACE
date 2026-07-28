# Radical experiment namespace

These suites implement the ordered gates in `docs/radical_model_plan.md`.
They reuse the existing local-validation split, renderer, and scorer through
`scripts/run_experiment_suite.py`; the radical adapter does not alter any of
those components.

Every adapter invocation creates:

```text
outputs/radical_runs/<timestamp>_<suite>_<stage>/
|-- manifest.json
|-- run.log
`-- suite_summary.json
```

The manifest records the Git state, Python/package/GPU environment, resolved
commands, metrics, render count, final Gaussian count when a checkpoint can be
inspected, peak device memory sampled with `nvidia-smi`, and the declared gate.

## R0

Inspect commands without training:

```bash
python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r0_control.json \
  --dry-run
```

Run both repeated controls:

```bash
python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r0_control.json \
  --stage full
```

R0 passes only when both render counts match the locked split/scorer and the
absolute score delta is no greater than `0.10`.

## R1

Refresh the editable package and run the synthetic unit tests first:

```bash
python -m pip install --force-reinstall --no-deps --editable .
python -m unittest discover -s tests -v
```

Run the smoke suite before the 20k pair:

```bash
python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r1_pixel_smoke.json \
  --stage train

python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r1_pixel_gs.json \
  --stage full
```

`P0` and `P1` use the same subclass, scene, seed, iteration budget, density
thresholds, loss, and appearance settings. Only P1 enables coverage-weighted
gradient accumulation and camera-depth scaling. R2 must not start unless
`P1 - P0 >= 0.20` on HCM0421.
