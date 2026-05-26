import os
import shutil
import numpy as np
from cmdstanpy import CmdStanModel
from stan_data_generator import make_stan_data
from io_helpers import save_metadata

ADAPT_DELTA = {"regression": 0.90, "classification": 0.80}
MODEL_DIR   = {"regression": "bnn_regression_models", "classification": "bnn_classification_models"}


def run_model(task, model_name, config_name, X_train, X_test, y_train, y_test, args):
    """Compile and sample a Stan BNN model for regression or classification.

    task: "regression" or "classification"
    """
    assert task in ("regression", "classification")

    if getattr(args, "seed", None) is not None:
        np.random.seed(args.seed)

    stan_data = make_stan_data(model_name, task, X_train, y_train, X_test, args)

    model_path = os.path.join(MODEL_DIR[task], f"{model_name}.stan")
    model = CmdStanModel(stan_file=model_path, force_compile=True)

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=args.samples,
        iter_warmup=args.burnin_samples,
        adapt_delta=ADAPT_DELTA[task],
        parallel_chains=4,
        show_console=False,
        max_treedepth=12,
    )

    output_dir = args.model_output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_metadata(output_dir, args, config_name)

    for i, path in enumerate(fit.runset.csv_files, start=1):
        shutil.copy(path, os.path.join(output_dir, f"chain_{i}.csv"))

    print(f"[✓] Saved results to: {output_dir}")
