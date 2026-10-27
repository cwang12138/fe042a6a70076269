# Reproduction

This package provides the files and commands for reproducing the included experiments on PEMS03 and PEMS08 with an input length of 96 and prediction lengths of 12, 24, 48, and 96. The required data, pretrained checkpoints, and evaluation checkpoints are included.

Run all commands from the repository root.

## Environment

Python 3.8–3.10 and an NVIDIA GPU with CUDA support are required.

```bash
python -m pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Training

Run both datasets with all four prediction lengths:

```bash
bash run_main.sh
```

Run one dataset:

```bash
bash scripts/MegaCRN/enhance/PEMS03.sh
bash scripts/MegaCRN/enhance/PEMS08.sh
```

The training scripts reuse the provided Stage 1 checkpoints by default. To retrain Stage 1:

```bash
STB_USE_PRETRAINED=0 bash scripts/MegaCRN/enhance/PEMS08.sh
```

To select a GPU and prediction lengths:

```bash
GPU=0 STB_Q_LIST="48 96" bash scripts/MegaCRN/enhance/PEMS08.sh
```

The default seeds are 100 for Stage 1 and 2025 for the subsequent training stage. New logs and checkpoints are saved under `experiments/MegaCRN_True/<dataset>/96-<prediction_length>/`.

## Evaluation

Evaluate the provided checkpoints without retraining:

```bash
bash scripts/MegaCRN/test/PEMS03.sh
bash scripts/MegaCRN/test/PEMS08.sh
```

The evaluation scripts report MAE, RMSE, MAPE, and inference timings. The `GPU` and `STB_Q_LIST` environment variables can also be used with these scripts.
