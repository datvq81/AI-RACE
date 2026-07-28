# Radical model execution status

Updated: 2026-07-28

## R0 - implemented, empirical gate pending

Implemented:

- isolated configs under `configs/experiments/radical`;
- isolated pinned container files `Dockerfile.radical` and
  `docker-compose.radical.yml`;
- adapter `scripts/run_radical_suite.py`, which delegates to the existing split,
  renderer, and scorer without modifying them;
- per-invocation manifest with Git/environment/command/log/metrics/render count,
  checkpoint Gaussian count, peak device memory, and gate evaluation;
- repeated D1b controls `R0C0` and `R0C1`.

Validated locally:

- both R0 commands resolve with identical scene, seed, iterations, loss,
  appearance, density, renderer, and evaluator settings;
- dry-run manifests are generated and correctly report an unavailable gate when
  render/metric artifacts do not exist.

Still required for the R0 pass:

```bash
python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r0_control.json \
  --stage full
```

The workspace used for implementation does not contain
`data/private_test1/HCM0421` or `data/local_validation/HCM0421`, and its host
Python does not have Nerfstudio/gsplat installed. Therefore no score or render
count has been fabricated.

## R1 - implemented, smoke and score gates pending

Sources reviewed:

- [Pixel-GS paper](https://arxiv.org/abs/2403.15530)
- [official Pixel-GS implementation](https://github.com/zhengzhang01/Pixel-GS)
- [Nerfstudio 1.1.4 Splatfacto](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.4/nerfstudio/models/splatfacto.py)

Implemented:

- `splatfacto-pixel` in the separate `var_nvs.radical` namespace;
- P0 exact parent path with stock equal-view AbsGrad averaging;
- P1 projected-footprint coverage weighting;
- Pixel-GS squared camera-depth scaling with `gamma_depth=0.37`;
- same-code P0/P1 20k suite and a smoke suite;
- synthetic unit tests for weighted averaging, distance scaling, footprint
  clipping/nonfinite inputs, and gate evaluation.

gsplat 1.0.0 does not expose the official modified rasterizer's exact count of
pixels that survive alpha/transmittance tests. The implementation therefore uses
the in-frame area derived from Nerfstudio's retained projected center/radius.
This is the plan's permitted screen-space area signal and does not change the
rasterizer, loss, density thresholds, or appearance model.

Validated locally:

```text
python -m unittest discover -s tests -v
Ran 8 tests ... OK
```

Still required, in order:

```bash
python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r1_pixel_smoke.json \
  --stage train

python scripts/run_radical_suite.py \
  --suite configs/experiments/radical/r1_pixel_gs.json \
  --stage full
```

R2 is intentionally not started. It is allowed only if the generated manifest
reports `P1 - P0 >= +0.20` on HCM0421.

## Empirical update - 2026-07-28

- R0 repeated-control gate passed.
- P1 smoke training completed.
- P0 score: `72.310275`, 3,173,337 Gaussians, 10,323 MiB peak device memory.
- P1 score: `72.468252`, 4,297,014 Gaussians, 14,051 MiB peak device memory.
- P1 improved all three component metrics, but its `+0.157978` score delta
  missed the required `+0.20` gate.
- The HCM0421 camera radius measured after the saved Nerfstudio dataparser
  transform and scale is `1.612979536`, rather than the initial placeholder
  `1.0`.
- `r1_pixel_gs_radius.json` defines P1R as a density-only retry using the
  measured radius and a unique checkpoint tag. R2 remains blocked until P1R
  passes the R1 gate.
