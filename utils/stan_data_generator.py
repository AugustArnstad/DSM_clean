import numpy as np


def make_stan_data(model_name, task, X_train, y_train, X_test, args):
    """Build a Stan data dict for the four paper models.

    Models: gaussian_tanh, regularized_horseshoe_tanh,
            dirichlet_horseshoe_tanh_nodewise, dirichlet_student_t_tanh_nodewise
    Tasks:  regression, classification
    """
    if task == "classification":
        y = y_train.astype(int)
    else:
        y = y_train.reshape(-1, 1)

    base = {
        "N":            X_train.shape[0],
        "P":            args.p,
        "X":            X_train,
        "y":            y,
        "L":            args.L,
        "H":            args.H,
        "N_test":       X_test.shape[0],
        "X_test":       X_test,
        "output_nodes": args.num_classes,
    }

    if model_name == "gaussian_tanh":
        return base

    if model_name == "regularized_horseshoe_tanh":
        return {**base, "p_0": 4, "a": 2.0, "b": 4.0}

    if model_name == "dirichlet_horseshoe_tanh_nodewise":
        return {**base, "p_0": 4, "a": 2.0, "b": 4.0,
                "alpha": 0.1 * np.ones(args.p), "gamma": 1e-3}

    if model_name == "dirichlet_student_t_tanh_nodewise":
        return {**base, "p_0": 4, "a": 2.0, "b": 4.0,
                "alpha": 0.1 * np.ones(args.p)}

    raise ValueError(f"Unknown model: '{model_name}'")
