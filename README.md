# RADD: Risk-Aware Distillation Defense Against Membership Inference Attacks on Iteratively Pruned Neural Networks

This repository contains the code and experiment scripts for our paper:

**RADD: Risk-Aware Distillation Defense Against Membership Inference Attacks on Iteratively Pruned Neural Networks**.

RADD is a pruning-time privacy defense for iteratively pruned neural networks. It targets a leakage channel that remains in memorization-aware pruning defenses: high-risk training samples can still impose sharp hard-label supervision during recovery fine-tuning. RADD treats the supervision target itself as a defense surface by combining:

- **High-risk exposure budgeting**: high-memorization-risk samples are replayed only under a strict exposure cap and ratio constraint.
- **Risk-aware teacher distillation**: high-risk hard-label learning is replaced with soft targets from the pre-pruning teacher model.
- **Confidence smoothing**: entropy regularization reduces overconfident high-risk predictions that are exploitable by membership inference attacks.

The implementation supports the main paper experiments on image and tabular benchmarks, including CIFAR100, CINIC, RESISC45, and Location, with ResNet18, VGG16BN, DenseNet121, MobileNetV2, and ColumnFC architectures.

## Repository Overview

```text
WeMeM-main/
  config/                         Dataset/model configuration files
  data/datasets/                   Local datasets
  memscore/                        Memorization-score CSV files
  result/                          Pretrained and pruned models
  log/                             Experiment logs and plotted figures

  pretrain_modi.py                 Train victim and shadow models
  generate_memscore_modi.py        Generate memorization scores
  prune_modi.py                    Prune and fine-tune models with defenses
  mia_modi.py                      Run membership inference attacks
  attackers.py                     Attack implementations
  datasets.py                      Dataset loaders
  data_process.py                  Tabular-data preprocessing
  base_model.py                    Model factory and architecture helpers
  pruner.py                        Pruning utilities

  run_memscore_batch.sbt           Batch pretraining and mem-score preparation
  run_baseline_defenses.sbt        Base and baseline defense experiments
  run_risk_distill_swmr.sbt        Main RADD defense experiments
  run_append_nn_attacks.sbt        Append NN-family attacks to existing logs
  run_lira_eval.sbt                Low-FPR LiRA evaluation
  run_radd_mechanism.sbt           Mechanism-analysis data extraction
  run_radd_runtime_overhead.sbt    Runtime-overhead experiments
  run_radd_kd_weighting.sbt        Distillation-weighting ablation
  run_risk_distill_ablation.sbt    RADD component ablation
  run_rd_pruning_way_fig8.sbt      Iterative vs. one-shot pruning study
  run_rd_structured_fig9.sbt       Structured pruning study
  run_radd_sparsity_sensitivity.sbt
  run_radd_window_robustness.sbt
  run_radd_fig13_attack_robustness.sbt

  plot_defense.py
  plot_lira.py
  plot_radd_mechanism.py
  plot_rd_pruning_way_fig8.py
  plot_rd_structured_fig9.py
  plot_radd_sparsity_sensitivity.py
  plot_radd_window_robustness.py
  plot_radd_fig13_attack_robustness.py
  plot_rd_sensitivity.py
```

The files with the `.sbt` suffix are SLURM batch scripts. They are ordinary Bash scripts containing `#SBATCH` directives. Before using them on a server, edit the `PROJECT_DIR`, `DATASETS`, `ARCHITECTURES`, and attack settings near the top of each script.

## Environment

The original experiments were run with Python 3.6.13 and PyTorch 1.10.1. A compatible setup is:

```bash
conda create -n radd python=3.6.13
conda activate radd

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install scikit-learn pandas matplotlib nni==2.3 pynvml tensorboard tqdm scipy
```

The provided `requirements.txt` records the environment used in our Linux experiments. If your CUDA or driver version differs, install the matching PyTorch build first and then install the remaining dependencies.

## Datasets

The paper focuses on the following datasets:

| Dataset | Type | Paper role | Architectures |
| --- | --- | --- | --- |
| CIFAR100 | Image | Fine-grained natural-image benchmark | ResNet18, VGG16BN, DenseNet121, MobileNetV2 |
| CINIC | Image | Broader natural-image distribution | ResNet18, VGG16BN, DenseNet121, MobileNetV2 |
| RESISC45 | Image | Remote-sensing scene classification | ResNet18, VGG16BN, DenseNet121, MobileNetV2 |
| Location | Tabular | Membership-inference tabular benchmark | ColumnFC |

The repository also contains configuration files for additional exploratory datasets. Those files are useful for extension experiments, but the paper claims should be interpreted according to the datasets reported in the manuscript.

Expected dataset locations:

```text
data/datasets/cifar100-data/
data/datasets/cinic/
data/datasets/resisc45/
data/datasets/location/
```

CIFAR100 can be downloaded automatically through `torchvision`. CINIC, RESISC45, and Location should be placed manually according to the corresponding loader requirements in `datasets.py` and `data_process.py`.

For tabular datasets, the train/test split must match the memorization-score file. If the split changes, regenerate the memorization scores before running RADD.

## Memorization Scores

RADD uses a memorization score `m_i` for each training sample. Scores are stored under `memscore/` and are consumed by `prune_modi.py` during risk-aware fine-tuning.

To prepare victim/shadow models and memorization scores in batch, edit and submit:

```bash
sbatch run_memscore_batch.sbt
```

Important settings in the script:

- `DATASETS`: datasets to process.
- `ARCHITECTURES`: model architectures to process.
- `K_FOLDS`: number of folds used for score estimation.
- `RUN_PRETRAIN_IF_MISSING`: whether to train missing victim/shadow models.
- `FORCE_PRETRAIN`: whether to overwrite stale pretrained models.

## Main RADD Command

A single RADD pruning/fine-tuning run can be launched with:

```bash
python prune_modi.py 0 ./config/cifar100_resnet18.json \
  --pruner_name iter_pruning \
  --prune_sparsity 0.6 \
  --prune_iter 5 \
  --defend risk_distill_swmr \
  --weight_decay_mem 0.01 \
  --stride 5 \
  --width 100 \
  --mem_thre 0.6 \
  --risk_gamma 1.0 \
  --distill_temp 3.0 \
  --distill_weight 1.0 \
  --hard_high_weight 0.0 \
  --entropy_weight 0.2 \
  --high_risk_cap 1 \
  --high_risk_ratio 0.1
```

For tabular experiments, use `iter_prunetxt` and a `columnfc` configuration such as `./config/location.json`.

## Membership Inference Attacks

RADD is evaluated against multiple MIA families:

| Attack option | Reported attack(s) |
| --- | --- |
| `threshold` | Conf, Entr, Mentr, Hconf |
| `samia` | SAMIA |
| `nn` | NN |
| `nn_top3` | Top3-NN |
| `nn_cls` | Cl-NN |
| `lira` | Low-FPR LiRA evaluation |

Run adaptive attacks on a defended model with:

```bash
python mia_modi.py 0 ./config/cifar100_resnet18.json \
  --pruner_name iter_pruning \
  --prune_sparsity 0.6 \
  --attacks threshold,samia,nn,nn_top3,nn_cls \
  --defend risk_distill_swmr \
  --adaptive
```

When computing **average attack accuracy** for the main paper setting, include all available attacks:

```text
Conf, Entr, Mentr, Hconf, SAMIA, NN, Top3-NN, Cl-NN
```

If an older experiment log does not contain NN, Top3-NN, and Cl-NN, do not silently claim an eight-attack average. Either append the missing attacks with `run_append_nn_attacks.sbt` or explicitly report that the average is computed over the available attack set.

## Recommended Reproduction Workflow

For a clean server reproduction, use the scripts in the following order.

### 1. Prepare data and configurations

Place datasets under `data/datasets/` and verify that the needed config files exist, for example:

```text
config/cifar100_resnet18.json
config/cifar100_vgg16.json
config/cifar100_dense.json
config/cifar100_mobilenetv2.json
config/cinic_resnet18.json
config/resisc45_resnet18.json
config/location.json
```

### 2. Generate or verify memorization scores

```bash
sbatch run_memscore_batch.sbt
```

This step prepares pretrained victim/shadow models and memorization scores.

### 3. Run baseline defenses

```bash
sbatch run_baseline_defenses.sbt
```

This script runs the undefended pruning baseline and comparison defenses such as PPB, ADV, DP, RelaxLoss, RSW, RMR, and SWMR, depending on the script settings.

### 4. Run RADD

```bash
sbatch run_risk_distill_swmr.sbt
```

This is the main RADD experiment script. It runs `risk_distill_swmr`, the implementation name for RADD in this codebase.

### 5. Append missing NN-family attacks if needed

```bash
sbatch run_append_nn_attacks.sbt
```

Use this step when earlier logs contain only threshold attacks and SAMIA. It helps avoid recomputing completed defenses while filling in NN, Top3-NN, and Cl-NN attack results.

### 6. Run specialized paper experiments

Submit the corresponding scripts as needed:

```bash
sbatch run_lira_eval.sbt
sbatch run_radd_mechanism.sbt
sbatch run_rd_pruning_way_fig8.sbt
sbatch run_rd_structured_fig9.sbt
sbatch run_radd_sparsity_sensitivity.sbt
sbatch run_radd_window_robustness.sbt
sbatch run_radd_fig13_attack_robustness.sbt
sbatch run_radd_kd_weighting.sbt
sbatch run_risk_distill_ablation.sbt
sbatch run_rd_sensitivity.sbt
sbatch run_radd_runtime_overhead.sbt
```

## Plotting

After synchronizing server logs back to the local machine, generate figures with:

```bash
python plot_defense.py
python plot_lira.py
python plot_radd_mechanism.py
python plot_rd_pruning_way_fig8.py
python plot_rd_structured_fig9.py
python plot_radd_sparsity_sensitivity.py
python plot_radd_window_robustness.py
python plot_radd_fig13_attack_robustness.py
python plot_rd_sensitivity.py
```

Most plots are saved under:

```text
log/<experiment_name>/figures/
```

The paper copies final PDF figures into the LaTeX project under:

```text
../IEEE_Template/fig/
```

## Output Files

Experiment outputs are written to `log/`. For example:

```text
log/cifar100_resnet18/iter_pruning_0.6_.txt
log/cifar100_resnet18/iter_pruning_0.6_risk_distill_swmr.txt
log/radd_mechanism/cifar100_resnet18/
log/radd_sparsity_sensitivity/
log/radd_window_robustness/
log/radd_fig13_attack_robustness/
```

Common fields in log files:

- `Victim pruned model test accuracy`: prediction accuracy of the pruned victim model.
- `<attack name> attack accuracy`: MIA attack accuracy.
- `Total <defense name> defend time`: defense runtime.

## Method Names in Code

Several defense names are inherited from the original pruning-privacy codebase:

| Code name | Paper name / role |
| --- | --- |
| empty defense name | Base iterative pruning |
| `slide` | RSW(H-to-L) |
| `slide_re` | RSW(L-to-H) |
| `ml2` | RMR |
| `slide_ml2` | SWMR(H-to-L) |
| `slide_re_ml2` | SWMR(L-to-H) |
| `risk_distill_swmr` | RADD |
| `ppb` | Prediction purification baseline |
| `adv` | Adversarial regularization |
| `dp` | Differential privacy baseline |
| `relaxloss` | RelaxLoss |

## Key Hyperparameters

| Parameter | Meaning | Default in main RADD experiments |
| --- | --- | --- |
| `--prune_sparsity` | Target pruning sparsity | `0.6` |
| `--prune_iter` | Number of pruning iterations | `5` |
| `--mem_thre` | Memorization-score threshold for high-risk samples | `0.6` |
| `--high_risk_cap` | Maximum high-risk exposure count | `1` |
| `--high_risk_ratio` | High-risk ratio in class-balanced window | `0.1` |
| `--distill_temp` | Distillation temperature | `3.0` |
| `--distill_weight` | KD loss weight | `1.0` |
| `--hard_high_weight` | Residual hard-label weight for high-risk samples | `0.0` |
| `--entropy_weight` | Entropy regularization weight | `0.2` for images, `0.1` for tabular data |
| `--weight_decay_mem` | Memorization-weighted regularization strength | `0.01` |
| `--risk_gamma` | Risk-weight exponent | `1.0` |

## Notes for Extending to New Datasets

To add a new dataset:

1. Add a dataset loader in `datasets.py` or tabular preprocessing in `data_process.py`.
2. Add a config file under `config/`.
3. Verify that `base_model.py` supports the intended architecture.
4. Generate memorization scores with `run_memscore_batch.sbt`.
5. Run Base, baselines, RADD, and the full attack suite.
6. Report average attack accuracy only over attacks that were actually executed.

For a dataset to be useful in this paper's evaluation style, the undefended pruned model should exhibit non-trivial membership leakage and the competing defenses should not already push nearly all attacks to random-guessing behavior without utility loss. Otherwise, RADD may be correct but visually indistinguishable from existing defenses.

## Acknowledgements

This repository builds on and extends code from prior work on membership inference attacks and defenses for pruned neural networks. Some inherited modules and baseline names follow the original implementation conventions. The RADD-specific extensions include risk-aware distillation, high-risk exposure budgeting, additional NN-family attack support, mechanism analysis, sensitivity studies, and paper-specific plotting scripts.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{radd2026,
  title   = {RADD: Risk-Aware Distillation Defense Against Membership Inference Attacks on Iteratively Pruned Neural Networks},
  author  = {Shigen Shen and Lanrui Yin and Yizhou Shen and Jingnan Dong and Wenlong Ke and Ruilong Deng and Tian Wang},
  journal = {Under Review},
  year    = {2026}
}
```

The citation block should be updated with the final author list and venue after publication.
