import os
import glob
from cmdstanpy import from_csv


# Maps display name -> (prior_subdir, posterior_subdir) under results_dir/
MODEL_PATHS = {
    # BNN models (paper Section 5)
    "Gaussian":              ("gaussian_tanh_prior",                    "gaussian_tanh"),
    "RHS":                   ("regularized_horseshoe_tanh_prior",       "regularized_horseshoe_tanh"),
    "DHS":                   ("dirichlet_horseshoe_tanh_nodewise_prior", "dirichlet_horseshoe_tanh_nodewise"),
    "DST":                   ("dirichlet_student_t_tanh_nodewise_prior", "dirichlet_student_t_tanh_nodewise"),
    # Linear regression models (paper Fig 3-4)
    "Linreg Gaussian":                    ("linreg_gaussian",                    "linreg_gaussian"),
    "Linreg Regularized Horseshoe scaled":("linreg_regularized_horseshoe_scaled", "linreg_regularized_horseshoe_scaled"),
    "Linreg Dirichlet Horseshoe":         ("linreg_dirichlet_horseshoe",          "linreg_dirichlet_horseshoe"),
    "Linreg Dirichlet Student T":         ("linreg_dirichlet_student_t",          "linreg_dirichlet_student_t"),
    "Linreg Beta Horseshoe":              ("linreg_beta_horseshoe",               "linreg_beta_horseshoe"),
    "Linreg Beta Student T":              ("linreg_beta_student_t",               "linreg_beta_student_t"),
}


def load_fit(model_glob_path):
    files = sorted(glob.glob(model_glob_path))
    if not files:
        print(f"[WARNING] No files matched: {model_glob_path}")
        return None
    return from_csv(path=files, method="sample")


def get_model_fits(config, results_dir, models=None, include_prior=True, include_posterior=True):
    """Load CmdStanPy fits for a given dataset config.

    config:      subfolder name, e.g. 'Friedman_N100_p10_sigma1.00_seed1'
    results_dir: root results directory, e.g. 'results/regression/single_layer/tanh/friedman'
    models:      subset of MODEL_PATHS keys to load; None loads all four
    """
    paths = MODEL_PATHS if models is None else {k: MODEL_PATHS[k] for k in models}
    fits = {}

    for name, (prior_dir, post_dir) in paths.items():
        entry = {}

        if include_prior:
            fit = load_fit(os.path.join(results_dir, prior_dir, config, "chain_*.csv"))
            if fit is not None:
                entry["prior"] = fit

        if include_posterior:
            fit = load_fit(os.path.join(results_dir, post_dir, config, "chain_*.csv"))
            if fit is not None:
                entry["posterior"] = fit

        if entry:
            fits[name] = entry

    return fits


def extract_weights(fit, var_name="W_1"):
    return fit.stan_variable(var_name)
