# SAMA: Sparse Adaptive Momentum Attack against Federated Recommender Systems and its Countermeasures

Source code for the paper: **Sparse Adaptive Momentum Attack against Federated Recommender Systems and its Countermeasures**.

This repository implements the SAMA attack (Sparse Adaptive Momentum Attack) and the InertiaDamp defense for federated recommendation.

## Method Overview

- **SAMA (Attack)**: An untargeted poisoning attack that exploits the momentum of item embedding updates across federated rounds. It pushes selected items along the EMA-accelerated displacement direction with adaptive strength control to degrade recommendation performance while evading detection.
- **InertiaDamp (Defense)**: A defense mechanism that detects malicious updates by measuring the cosine similarity of per-item gradient directions between consecutive rounds. Items with abnormally high directional consistency (inertia) are dampened before aggregation.

## Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- NumPy

```bash
pip install -r requirements.txt
```

## Quick Start

Run attack and defense experiments with one command:

```bash
bash run.sh
```

Run SAMA attack (Sum aggregation, no defense):

```bash
python main.py --dataset ml-100k/ --agg_type Sum
```

Run with InertiaDamp defense:

```bash
python main.py --dataset ml-100k/ --agg_type InertiaDamp
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ml-100k/` | Dataset name under `Data/` |
| `--dim` | `32` | Latent vector dimension |
| `--lr` | `0.01` | Learning rate |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `1024` | Client batch size |
| `--agg_type` | `Sum` | Aggregation: `Sum` (no defense), `InertiaDamp` (our defense) |
| `--grad_limit` | `1.0` | L2-norm clip threshold for gradients |
| `--device` | `cuda`/`cpu` | Device (auto-detected) |

## Dataset

The repository includes the **ml-100k** dataset under `Data/ml-100k/`. Format: each line is `user_id item1 item2 item3 ...`.

## Project Structure

```
├── main.py              # Entry point
├── parse.py             # Argument parsing
├── data.py              # Data loading
├── eval.py              # HR@K and NDCG@K metrics
├── run.sh               # One-click experiment script
├── requirements.txt
├── README.md
├── Data/
│   └── ml-100k/
│       ├── train.txt
│       └── test.txt
└── FedRec/
    ├── __init__.py
    ├── client.py              # Benign federated client (BPR training)
    ├── malicious_client.py    # SAMA attack client
    └── server.py              # Federated server (Sum + InertiaDamp defense)
```
