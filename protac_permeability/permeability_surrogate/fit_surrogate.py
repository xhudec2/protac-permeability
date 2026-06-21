from argparse import ArgumentParser

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from protac_permeability.chem_utils import calculate_properties, canonicalize_smiles
from protac_permeability.permeability_surrogate import (
    EnsemblePermeabilitySurrogate,
    PermeabilitySurrogate,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_SEED = 42


def fit_ensemble(data_path: str, save_dir: str, n_models: int, num_repeats: int = 5):
    df = pd.read_csv(data_path)
    smiles_list = df["SMILES"].tolist()
    smiles = [canonicalize_smiles(smi) for smi in smiles_list]
    descriptors = [calculate_properties(smi) for smi in smiles]
    X = np.stack(descriptors)
    y = df["PAMPA"].values
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    models = []
    sp = []
    for repeat in range(num_repeats):
        current_seed = RANDOM_SEED + repeat
        cv = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=current_seed)
        for itrain, itest in cv.split(smiles_list, y_bins):
            train_X = X[itrain]
            train_y = y[itrain]
            test_X = X[itest]
            test_y = y[itest]

            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                inner_cv = StratifiedKFold(
                    n_splits=2, shuffle=True, random_state=current_seed
                )
                preds, trues = [], []
                for itrain_inner, ival_inner in inner_cv.split(train_X, y_bins[itrain]):
                    train_X_inner = train_X[itrain_inner]
                    y_in_tr = train_y[itrain_inner]
                    val_X_inner = train_X[ival_inner]
                    y_in_val = train_y[ival_inner]

                    model = PermeabilitySurrogate(model=Ridge(alpha=alpha))
                    model.fit(train_X_inner, y_in_tr)
                    predictions = model.predict(val_X_inner)
                    preds.extend(predictions)
                    trues.extend(y_in_val)
                spearman_r = spearmanr(trues, preds)[0]
                return spearman_r

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=100)
            best_alpha = study.best_params["alpha"]
            model = PermeabilitySurrogate(model=Ridge(alpha=best_alpha))
            model.fit(train_X, train_y)
            models.append(model)
            spearman_corr = spearmanr(test_y, model.predict(test_X))[0]
            sp.append(spearman_corr)
    print(f"Mean Spearman correlation: {np.mean(sp):.4f} ± {np.std(sp):.4f}")
    ensemble_model = EnsemblePermeabilitySurrogate(models=models)
    ensemble_model.save(save_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fit a permeability surrogate model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing SMILES and permeability data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/ensemble",
        help="Path to save the fitted model (only used for single model).",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=5,
        help="Number of models to fit in the ensemble (only used for ensemble).",
    )
    args = parser.parse_args()

    fit_ensemble(
        data_path=args.data_path,
        save_dir=args.save_dir,
        n_models=args.n_models,
    )
