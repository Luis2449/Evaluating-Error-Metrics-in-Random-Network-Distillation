# Evaluating Error Metrics for Curiosity-Driven Exploration in Random Network Distillation

This repository contains the implementation used for the bachelor thesis:

**Evaluating Error Metrics for Curiosity-Driven Exploration in Random Network Distillation**

The thesis investigates how the prediction-error metric used by Random Network Distillation (RND) affects:

- exploration performance,
- intrinsic-reward stability,
- robustness to noisy observations.

The standard mean squared error (MSE) used by RND serves as the baseline. The experiments compare it with Huber loss, Reverse Huber loss, Cauchy loss, and fourth-power error.

## Based On

This implementation is adapted from:

[jcwleo/random-network-distillation-pytorch](https://github.com/jcwleo/random-network-distillation-pytorch)

The original MIT license is retained in [LICENSE](LICENSE).

## Evaluated Prediction Losses

The loss used for RND is selected through `RNDLoss` in `config.conf`.

The thesis evaluates the following values:

| Configuration value | Loss |
| --- | --- |
| `mse` | Mean squared error |
| `huber` | Huber loss |
| `reverse_huber` | Reverse Huber loss |
| `cauchy` | Cauchy loss |
| `e4` | Fourth-power error |

The selected loss is used both for training the RND predictor and for calculating the intrinsic reward.

## Experimental Environments

The thesis evaluates the prediction losses in two environments.

### Montezuma's Revenge

- Environment: `MontezumaRevengeNoFrameskip-v4`
- Parallel environments: 128
- Rollout length: 128 steps per environment
- Maximum training rollouts: 2500
- Maximum episode length: 4500 interaction steps
- Sticky-action probability: 0.25

### MiniGrid DoorKey

- Environment: `MiniGrid-DoorKey-8x8-v0`
- Parallel environments: 64
- Rollout length: 128 steps per environment
- Maximum training rollouts: 2500
- Maximum episode length: 4500 interaction steps
- Sticky-action probability: 0.25

Both environments use processed grayscale observations with a resolution of `84 x 84`. The PPO policy receives the four most recent processed frames, while RND receives the latest processed frame.

## Observation-Noise Experiment

Robustness to noisy observations is evaluated in MiniGrid DoorKey.

The noisy condition applies zero-mean Gaussian noise to every processed observation frame with:

- relative standard deviation: `0.05`
- absolute standard deviation: `12.75`
- corruption probability: `1.0`
- pixel range after clipping: `[0, 255]`
- noise target: policy and RND
- application point: after observation preprocessing

All other training and environment parameters are kept unchanged relative to the clean DoorKey condition.

## Experiment Configurations

The `configs/` directory contains the three main configurations corresponding to the thesis experiments:

```text
configs/
├── montezuma_clean.conf
├── doorkey_clean.conf
└── doorkey_noisy.conf
```

The training and evaluation programs read their settings from `config.conf`.

To select a configuration, copy the desired configuration file to `config.conf`.

For example, for clean Montezuma's Revenge:

```bash
cp configs/montezuma_clean.conf config.conf
```

For clean DoorKey:

```bash
cp configs/doorkey_clean.conf config.conf
```

For noisy DoorKey:

```bash
cp configs/doorkey_noisy.conf config.conf
```

Then set `RNDLoss` in `config.conf` to the desired prediction loss:

```text
RNDLoss = mse
```

The prediction losses evaluated in the thesis are:

```text
mse
huber
reverse_huber
cauchy
e4
```

## Main Training Parameters

The main optimization parameters are shared across the evaluated loss conditions.

| Parameter | Montezuma's Revenge | MiniGrid DoorKey |
| --- | ---: | ---: |
| Parallel environments | 128 | 64 |
| Steps per rollout and environment | 128 | 128 |
| Maximum rollout updates | 2500 | 2500 |
| PPO epochs per rollout | 4 | 4 |
| Minibatches per epoch | 4 | 4 |
| Learning rate | `1e-4` | `1e-4` |
| Extrinsic discount factor | `0.999` | `0.999` |
| Intrinsic discount factor | `0.99` | `0.99` |
| GAE parameter | `0.95` | `0.95` |
| PPO clipping parameter | `0.1` | `0.1` |
| Entropy coefficient | `0.001` | `0.001` |
| Intrinsic advantage coefficient | `1.0` | `1.0` |
| Extrinsic advantage coefficient | `2.0` | `2.0` |
| RND predictor update proportion | `0.25` | `0.25` |
| Observation-normalization warm-up | 50 rollout lengths | 50 rollout lengths |

## Environment Setup

A Conda environment definition is provided in `environment.yml`.

Create the environment with:

```bash
conda env create -f environment.yml
```

Activate it with:

```bash
conda activate rnd-3.9
```

For the Atari experiments, the required Atari ROMs must also be available to the Arcade Learning Environment.

## Training

After selecting the desired configuration and prediction loss, start training with:

```bash
python train.py
```

Training generates model checkpoints and TensorBoard logs under `models/`.

Generated models, checkpoints, logs, and other training outputs are intentionally excluded from the submission repository.

## Evaluation

Evaluation is run with:

```bash
python eval.py
```

Evaluation data is written to `eval_results/`.

Generated evaluation results and gameplay videos are intentionally excluded from the submission repository.

## Plotting Training Metrics

TensorBoard scalar data can be converted into plots using:

```bash
python make_animation.py --models-dir models --output-dir eval_results
```

## Repository Structure

The main files in the repository are:

```text
.
├── agents.py
├── config.conf
├── config.py
├── configs/
│   ├── doorkey_clean.conf
│   ├── doorkey_noisy.conf
│   └── montezuma_clean.conf
├── environment.yml
├── envs.py
├── eval.py
├── LICENSE
├── make_animation.py
├── model.py
├── obs_noise.py
├── README.md
├── tests/
├── train.py
└── utils.py
```

### Main Files

- `agents.py` — PPO/RND agent and prediction-loss implementations
- `envs.py` — environment wrappers, preprocessing, and environment-specific evaluation metadata
- `model.py` — neural-network architectures
- `obs_noise.py` — observation-noise implementation
- `train.py` — training procedure
- `eval.py` — evaluation procedure
- `utils.py` — shared utilities
- `config.py` — configuration loading
- `config.conf` — active experiment configuration
- `configs/` — configurations corresponding to the thesis experiments
- `tests/` — implementation tests
- `make_animation.py` — TensorBoard scalar plotting utility

## Generated Files

The following generated data is intentionally not included in the submission repository:

- trained model checkpoints,
- previous experiment results,
- evaluation outputs,
- TensorBoard event files,
- gameplay videos,
- runtime seed files.

These files are excluded through `.gitignore`.

## License

See [LICENSE](LICENSE).