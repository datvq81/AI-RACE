# Experiment suites

`scripts/run_experiment_suite.py` reads a JSON suite and invokes
`scripts/run_local_validation.py` once per enabled experiment. Jobs are
sequential, so one suite is safe to use on one GPU.

## Minimal schema

```json
{
  "schema_version": 1,
  "name": "my_search",
  "defaults": {
    "scene": "HCM0421",
    "method": "splatfacto-big",
    "iterations": 10000,
    "seed": 42,
    "train_args": [],
    "runner_args": []
  },
  "experiments": [
    {
      "id": "B1",
      "enabled": true,
      "description": "Short human-readable note",
      "method": "splatfacto-big",
      "tag": "B1_{budget}{seed_suffix}",
      "train_args": ["--pipeline.model.sh-degree", "2"]
    }
  ]
}
```

Per-experiment values override `defaults`. `train_args` and `runner_args` are
appended to their corresponding default lists. Supported tag placeholders are
`{id}`, `{scene}`, `{iterations}`, `{budget}`, `{seed}`, and `{seed_suffix}`.
`{seed_suffix}` is empty for seed 42 and `_sN` for another seed, preserving the
names of the original A1-A3b runs while preventing collisions for new seeds.

`method` is the Nerfstudio command name. It can therefore select either a stock
method such as `splatfacto-big` or a registered custom architecture such as
`thin-splatfacto`. Architecture-specific CLI flags belong in `train_args`.
The method/plugin must already be installed and visible in `ns-train --help`.
The single-scene runner injects its historical SH-degree and scale-regularizer
defaults only for stock method names `splatfacto` and `splatfacto-*`; custom
methods receive only their explicitly declared `train_args`.

## Commands

```bash
# Resolve and inspect commands without training
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --list
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --dry-run

# Train all enabled experiments, or a subset
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --stage train
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --stage train --only A2,A3b

# Render and score checkpoints that already exist
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --stage eval

# Train, render, and score each experiment before starting the next one
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --stage full

# Override a suite without editing it
python scripts/run_experiment_suite.py --suite configs/experiments/a_baseline.json --scene chair --iterations 20000 --seed 7
```

The legacy wrapper accepts the same CLI options plus environment overrides:

```bash
SCENE=HCM0421 ITERATIONS=10000 SEED=42 bash scripts/train_a_configs.sh --only A2,A3b
SUITE=configs/experiments/my_search.json STAGE=full bash scripts/train_a_configs.sh
```

Do not reuse a tag after changing its effective method or arguments. Choose a
new experiment ID/tag, otherwise `run_local_validation.py` intentionally reuses
the completed checkpoint under the old name. Use `--new-run` only when a
timestamped duplicate is actually wanted.

## Custom edge-loss method

The repo provides `splatfacto-edge`, a package-owned extension of Nerfstudio
1.1.4's `splatfacto-big`. It adds a normalized RGB Sobel-gradient L1 loss and
does not modify `site-packages`:

```bash
python -m pip install --no-deps --editable .
python -m nerfstudio.scripts.train splatfacto-edge --help
```

The model options are:

- `--pipeline.model.edge-loss-weight`: non-negative loss weight; zero is the
  control behavior.
- `--pipeline.model.edge-loss-start-step`: first active step, default `0`.
- `--pipeline.model.edge-loss-end-step`: first inactive step; `-1` means the
  loss remains active.

Run the prepared control and weight ablation with:

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/c_edge_loss.json \
  --stage full \
  --scene HCM0421 \
  --iterations 20000 \
  --seed 42
```

`C0` validates that the subclass with weight zero matches the A2 family. `C1a`
uses weight `0.02`; `C1b` uses weight `0.05`. Compare C1a/C1b against C0 from
the same suite because CUDA rasterization is not perfectly deterministic even
when all effective options and the seed are identical.

## Full-image LPIPS training method

The repo also provides `splatfacto-perceptual`. It reuses Splatfacto's frozen
Alex-LPIPS network and adds its differentiable result to the training loss over
the complete rendered image. It does not crop patches or load a second AlexNet.

The model options are:

- `--pipeline.model.lpips-loss-weight`: non-negative LPIPS loss weight; zero is
  the control behavior.
- `--pipeline.model.lpips-loss-start-step`: first active step, default `6000`.
  Splatfacto reaches full training resolution at this step with its standard
  `num_downscales=2` and `resolution_schedule=3000` settings.
- `--pipeline.model.lpips-loss-end-step`: first inactive step; `-1` keeps it
  active until training ends.

Install or refresh the editable method registration on an existing pod:

```bash
cd /workspace/AI-RACE/var2026-digital-twin
python -m pip install --no-deps --editable .
python -m nerfstudio.scripts.train splatfacto-perceptual --help >/dev/null
```

Then run the 20k control and weight screen:

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/d_full_image_lpips.json \
  --stage full \
  --scene HCM0421 \
  --iterations 20000 \
  --seed 42 \
  2>&1 | tee D_HCM0421_20k.log
```

`D0` disables LPIPS, `D1a` uses weight `0.01`, and `D1b` uses weight `0.03`.
Compare D1a/D1b with D0 from this suite. Only promote a candidate if its gain
is clearly larger than the small run-to-run noise observed in earlier controls.

## Directional SH background

The `splatfacto-sky` method keeps the D1b foreground and LPIPS objective, but
renders the Gaussians over black and fills residual transmittance with a tiny
learned RGB function of world-space ray direction. Degree three adds only 48
trainable scalar coefficients and has negligible VRAM cost.

The relevant model options are:

- `--pipeline.model.use-directional-background`: enable the learned background.
- `--pipeline.model.background-sh-degree`: SH degree from `0` through `3`.
- `--pipeline.model.background-start-step`: delay background learning so early
  geometry cannot immediately delegate all sky pixels to the background.
- `--pipeline.model.background-init-color`: initial RGB tuple; the suite leaves
  this at neutral gray.

Run D2 alone, based on the winning D1b settings:

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/d_directional_background.json \
  --stage full \
  --scene HCM0421 \
  --iterations 20000 \
  --seed 42 \
  --only D2
```

The same suite also contains `D0` (directional background disabled) and `D1`
(SH degree 2) if a same-subclass control or degree ablation is needed later.

## Staged MSE fine-tuning

The `splatfacto-staged` method extends the winning D1b full-image LPIPS model.
Before the configured transition it leaves Splatfacto's main loss unchanged.
After the transition, `staged_mse_weight` is transferred from L1 to MSE:

```text
before 15k: 0.80 L1 + 0.20 DSSIM + 0.03 LPIPS (LPIPS starts at 6k)
after  15k: 0.45 L1 + 0.35 MSE + 0.20 DSSIM + 0.03 LPIPS
```

The model options are:

- `--pipeline.model.staged-mse-weight`: weight transferred from L1 to MSE.
- `--pipeline.model.staged-mse-start-step`: first active MSE step.
- `--pipeline.model.staged-mse-end-step`: first inactive step; `-1` keeps it
  active until training ends.

Run the same-subclass control and F1 at 20k:

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/f_staged_mse.json \
  --stage full \
  --scene HCM0421 \
  --iterations 20000 \
  --seed 42 \
  --only F0,F1
```

Run only F1 when the existing D1b result is being used as the control:

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/f_staged_mse.json \
  --stage full \
  --scene HCM0421 \
  --iterations 20000 \
  --seed 42 \
  --only F1
```
