import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import timesnet_train
import A_train


DATA_DEFAULTS = {
    "kdd": {"seq_len": 48, "enc_in": 99, "c_out": 99},
    "guangzhou": {"seq_len": 48, "enc_in": 214, "c_out": 214},
    "physio": {"seq_len": 48, "enc_in": 40, "c_out": 40},
}

MODEL_REGISTRY = {
    "timesnet": (timesnet_train.diffusion_train, timesnet_train.diffusion_test),
    "fgti": (A_train.diffusion_train, A_train.diffusion_test),
}


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch: int = 16
    dataset: str = "kdd"
    model: str = "timesnet"
    missing_rate: float = 0.2
    seed: int = 3407
    seq_len: int = 48
    enc_in: int = 99
    c_out: int = 99
    epoch_diff: int = 10
    learning_rate_diff: float = 1e-3
    task_name: str = "imputation"
    d_model: int = 128
    e_layers: int = 2
    d_ff: int = 2048
    n_heads: int = 8
    d_layers: int = 1
    top_k: int = 5
    num_kernels: int = 6
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.1
    pred_len: int = 0
    label_len: int = 0
    num_class: int = 0
    data_root: str = str(Path(__file__).resolve().parents[1] / "Datasets" / "data")
    results_root: str = str(Path(__file__).resolve().parents[1] / "Results")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified training entry")
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--batch", type=int, default=Config.batch)
    p.add_argument("--dataset", type=str, default=Config.dataset, choices=list(DATA_DEFAULTS.keys()))
    p.add_argument("--model", type=str, default=Config.model, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--missing_rate", type=float, default=Config.missing_rate, help="e.g. 0.2/0.4/0.6")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--enc_in", type=int, default=None)
    p.add_argument("--c_out", type=int, default=None)
    p.add_argument("--epoch_diff", type=int, default=Config.epoch_diff)
    p.add_argument("--learning_rate_diff", type=float, default=Config.learning_rate_diff)
    p.add_argument("--task_name", type=str, default=Config.task_name)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--e_layers", type=int, default=Config.e_layers)
    p.add_argument("--d_ff", type=int, default=Config.d_ff)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    p.add_argument("--d_layers", type=int, default=Config.d_layers)
    p.add_argument("--top_k", type=int, default=Config.top_k)
    p.add_argument("--num_kernels", type=int, default=Config.num_kernels)
    p.add_argument("--embed", type=str, default=Config.embed)
    p.add_argument("--freq", type=str, default=Config.freq)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--pred_len", type=int, default=Config.pred_len)
    p.add_argument("--label_len", type=int, default=Config.label_len)
    p.add_argument("--num_class", type=int, default=Config.num_class)
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--results_root", type=str, default=Config.results_root)
    return p


def apply_dataset_defaults(cfg):
    defaults = DATA_DEFAULTS[cfg.dataset]
    if cfg.seq_len is None:
        cfg.seq_len = defaults["seq_len"]
    if cfg.enc_in is None:
        cfg.enc_in = defaults["enc_in"]
    if cfg.c_out is None:
        cfg.c_out = defaults["c_out"]


def main():
    parser = build_parser()
    cfg = parser.parse_args()
    apply_dataset_defaults(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    train_fn, test_fn = MODEL_REGISTRY[cfg.model]

    print(f"[Run] model={cfg.model} dataset={cfg.dataset} miss={cfg.missing_rate} seed={cfg.seed}")
    model = train_fn(cfg)
    test_fn(cfg, model)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
