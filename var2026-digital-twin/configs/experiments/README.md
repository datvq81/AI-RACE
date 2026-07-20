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
