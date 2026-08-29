from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)

from protac_permeability.chem_utils import (
    butina_groups,
    calculate_properties,
    canonicalize_smiles,
)
from protac_permeability.permeability_surrogate import (
    EnsemblePermeabilitySurrogate,
    PermeabilitySurrogate,
)

RANDOM_SEED = 42


def save_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split: int,
    save_dir: str,
) -> None:
    splits_dir = Path(save_dir)
    (splits_dir / "train").mkdir(parents=True, exist_ok=True)
    (splits_dir / "test").mkdir(parents=True, exist_ok=True)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_df.to_csv(f"{save_dir}/train/split_{split}.csv", index=False)
    test_df.to_csv(f"{save_dir}/test/split_{split}.csv", index=False)


def get_splits_butina(
    smiles: np.ndarray,
    source: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    current_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=current_seed
    )
    yield from cv.split(smiles, source, groups)


def get_splits(
    smiles: np.ndarray,
    source: np.ndarray,
    n_splits: int,
    current_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
    yield from cv.split(smiles, source)


def fit_ensemble(
    data_path: str,
    save_dir: str,
    n_models: int,
    descriptor_cols: None | list[str] = None,
    num_repeats: int = 5,
    original_only: bool = False,
    new_only: bool = False,
    dist_threshold: None | float = None,
    split_save_dir: str = None,
) -> list[list[float]]:
    df = pd.read_csv(data_path)
    smiles = np.array([canonicalize_smiles(smi) for smi in df["SMILES"]])

    if descriptor_cols is None:
        descriptors = [calculate_properties(smi) for smi in smiles]
        X = np.stack(descriptors)
    else:
        X = df[descriptor_cols].values

    y = df["logPAMPA"].values
    models = []
    preds = np.zeros((num_repeats, len(y)))
    source = np.where(df.protac_db_id.isna(), "new", "original")
    for repeat in range(num_repeats):
        current_seed = RANDOM_SEED + repeat
        if dist_threshold is not None:
            groups, _ = butina_groups(smiles, threshold=dist_threshold)
            splits_generator = get_splits_butina(
                smiles, source, groups, n_splits=n_models, current_seed=current_seed
            )
        else:
            splits_generator = get_splits(
                smiles, source, n_splits=n_models, current_seed=current_seed
            )

        for itrain, itest in splits_generator:
            train_X = X[itrain]
            train_y = y[itrain]
            test_X = X[itest]
            original_mask = ~df.protac_db_id.isna()

            if original_only:
                train_X = train_X[original_mask.values[itrain]]
                train_y = train_y[original_mask.values[itrain]]
            elif new_only:
                train_X = train_X[~original_mask.values[itrain]]
                train_y = train_y[~original_mask.values[itrain]]

            model = PermeabilitySurrogate(model=Ridge())

            model.fit(train_X, train_y)
            models.append(model)
            preds[repeat, itest] = model.predict(test_X)
            if split_save_dir is not None:
                save_split(
                    df,
                    train_idx=itrain,
                    test_idx=itest,
                    split=len(models) - 1,
                    save_dir=split_save_dir,
                )

    sps = [spearmanr(y, preds[i])[0] for i in range(num_repeats)]
    rmses = [np.sqrt(mean_squared_error(y, preds[i])) for i in range(num_repeats)]
    r2 = [r2_score(y, preds[i]) for i in range(num_repeats)]
    print(f"Mean Spearman correlation: {np.mean(sps):.2f} ± {np.std(sps):.2f}")
    print(f"Mean RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"Mean R2: {np.mean(r2):.2f} ± {np.std(r2):.2f}")
    ensemble_model = EnsemblePermeabilitySurrogate(models=models)
    ensemble_model.save(save_dir)
    return [
        [np.mean(sps), np.std(sps)],
        [np.mean(rmses), np.std(rmses)],
        [np.mean(r2), np.std(r2)],
    ]


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
        default="models/ensemble_final",
        help="Path to save the fitted model (only used for single model).",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=5,
        help="Number of models to fit in the ensemble (only used for ensemble).",
    )
    parser.add_argument(
        "--split_save_dir",
        type=str,
        default=None,
        help="Path to save the split data.",
    )
    parser.add_argument(
        "--dist_threshold",
        type=float,
        default=None,
        help="Threshold for butina clustering.",
    )
    parser.add_argument(
        "--original_only",
        action="store_true",
        help="Only use original data for training.",
    )
    parser.add_argument(
        "--new_only",
        action="store_true",
        help="Only use new data for training.",
    )
    parser.add_argument(
        "--descriptor_cols",
        default=None,
        nargs="+",
        help="Descriptor columns to use for training. If not provided, will calculate descriptors from SMILES.",
    )

    args = parser.parse_args()

    fit_ensemble(
        data_path=args.data_path,
        save_dir=args.save_dir,
        n_models=args.n_models,
        dist_threshold=args.dist_threshold,
        original_only=args.original_only,
        new_only=args.new_only,
        split_save_dir=args.split_save_dir,
        descriptor_cols=args.descriptor_cols,
    )
