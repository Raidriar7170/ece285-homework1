# ECE 285 Homework 1: Chest X-ray VAE

PyTorch implementation of a convolutional variational autoencoder trained on the assigned Kaggle Chest X-Ray Images (Pneumonia) dataset.

- **[Submission report (2-page PDF)](homework1/submission/Homework1_report.pdf)** — includes this repository URL.
- **[Setup and reproduction instructions](homework1/README.md)**
- **[Model](homework1/model.py)** · **[Training](homework1/train.py)** · **[Evaluation](homework1/evaluate.py)** · **[Generation](homework1/generate.py)**
- **[Observed metrics](homework1/runs/vae_cpu/metrics.json)** · **[Training history](homework1/runs/vae_cpu/history.json)**
- **[NeurIPS LaTeX source](homework1/report/paper/main.tex)**

## Recorded experiment

30 epochs on CPU; validation-selected epoch 26. FID (2048-dimensional torchvision Inception features): **196.60**. Inception Score: **1.769**. Evaluation uses 624 real test images and 624 independent prior samples. These are one-run measurements, not clinical validation; generated images are visibly blurry.

The dataset is not redistributed. Use `prepare_data.py` to download the assigned public Kaggle archive. The selected checkpoint and run records are provided under `homework1/runs/vae_cpu/`. See the report for preprocessing, metric conventions, and limitations.

The assignment accepts PDF only: submit the linked report; the code is accessible through its GitHub hyperlink.
