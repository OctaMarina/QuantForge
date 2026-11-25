"""Meta-model specifications, purged hyper-parameter search and probability calibration (needs scikit-learn).

    from quant_forge.models import default_model_specs, tune_purged, fit_predict, PlattCalibrator

    specs = default_model_specs(seed=0)                       # {"LR", "RF", "HGB"} with reproducible grids
    best = tune_purged(specs["LR"], X, y, folds)              # OOF log loss over the grid
    p_test, model, _ = fit_predict(specs["LR"], best.params, X, y, X_test)
    p_test = PlattCalibrator().fit(best.oof, y).predict(p_test)
"""

from .calibration import PlattCalibrator, logit
from .factories import (ModelSpec, default_model_specs, hgb_spec, lr_spec, make_hgb, make_lr, make_rf,
                        median_imputer, pca_preprocessor, positive_proba, rf_spec)
from .tuning import TuningResult, fit_predict, oof_prior, tune_purged

__all__ = ["ModelSpec", "default_model_specs", "lr_spec", "rf_spec", "hgb_spec", "make_lr", "make_rf", "make_hgb",
           "median_imputer", "pca_preprocessor", "positive_proba", "TuningResult", "tune_purged", "fit_predict",
           "oof_prior", "PlattCalibrator", "logit"]
