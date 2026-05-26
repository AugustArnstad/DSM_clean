"""Run one BNN model (regression or classification) across a dataset directory.

Usage:
  python run_models.py --task regression --model dirichlet_horseshoe_tanh_nodewise --H 16 --L 1
  python run_models.py --task regression --model gaussian_tanh --H 32 --L 2

Output layout:
  results/<model>_H<H>_L<L>/<config>/chain_*.csv

Dataset layout expected:
  Regression:    datasets/friedman/<name>_N{N}_p{p}_sigma{s}_seed{seed}.npz
  Classification: datasets/moons/<name>_N{N}_p{p}_sigma{s}_seed{seed}.npz

CmdStan path: set CMDSTAN env var on the remote, or let CmdStanPy find it automatically.
"""

import os
import re
import argparse
import numpy as np
from cmdstanpy import CmdStanModel
from model_runner import run_model

DATA_DIR = {
    "regression":     "datasets/friedman",
    "classification": "datasets/moons",
}

NUM_CLASSES = {
    "regression":     1,
    "classification": 2,
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",      required=True, choices=["regression", "classification"])
parser.add_argument("--model",     required=True, help="Stan model name without .stan")
parser.add_argument("--H",         type=int, default=16,   help="Hidden units per layer")
parser.add_argument("--L",         type=int, default=1,    help="Number of hidden layers")
parser.add_argument("--data_dir",  default=None,  help="Override default dataset directory")
parser.add_argument("--output_dir",default="results", help="Root output directory")
parser.add_argument("--pattern",   default=".*",  help="Regex to filter dataset filenames")
parser.add_argument("--N",         type=int, nargs="*", help="Filter by sample size(s)")
parser.add_argument("--sigma",     type=float, nargs="*", help="Filter by noise level(s)")
parser.add_argument("--warmup",    type=int, default=1000)
parser.add_argument("--sample",    type=int, default=1000)
parser.add_argument("--limit",     type=int, default=None, help="Max datasets to run")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

data_dir = args.data_dir or DATA_DIR[args.task]

files = sorted(
    f for f in os.listdir(data_dir)
    if f.endswith(".npz") and re.match(args.pattern, f)
)

processed = 0
for fname in files:
    tokens = fname.replace(".npz", "").split("_")
    data_type = tokens[0]
    N     = int(tokens[1][1:])
    p     = int(tokens[2][1:])
    sigma = float(tokens[3][5:])
    seed  = int(tokens[4][4:])

    if args.N     and N     not in args.N:     continue
    if args.sigma and sigma not in args.sigma: continue
    if args.limit is not None and processed >= args.limit: break

    config_name     = fname.replace(".npz", "")
    arch_tag        = f"{args.model}_H{args.H}_L{args.L}"
    model_output_dir = os.path.join(args.output_dir, arch_tag, config_name)

    if not args.overwrite and os.path.exists(model_output_dir):
        print(f"[Skip] {config_name}")
        continue

    data = np.load(os.path.join(data_dir, fname))
    print(f"[Run] {fname}")

    run_model(
        task=args.task,
        model_name=args.model,
        config_name=config_name,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        args=argparse.Namespace(
            p=p,
            H=args.H,
            L=args.L,
            num_classes=NUM_CLASSES[args.task],
            model=args.model,
            data=data_type,
            config=config_name,
            seed=seed,
            model_output_dir=model_output_dir,
            burnin_samples=args.warmup,
            samples=args.sample,
            standardize=True,
        ),
    )
    processed += 1
