import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler

from protac_permeability.chem_utils import (
    calculate_properties,
    canonicalize_smiles,
    descriptor_functions,
)

DEFAULT_DESCRIPTORS = descriptor_functions.copy()
DEFAULT_DESCRIPTORS.pop("CharVol")


class PermeabilitySurrogate:
    def __init__(self, model, scaler: StandardScaler | None = None):
        self.model = model
        self.scaler = scaler

    def fit(self, X: str | list[str] | np.ndarray, targets: np.ndarray):
        if not isinstance(X, np.ndarray):
            if isinstance(X, str):
                X = [X]
            smiles = [canonicalize_smiles(smi) for smi in X]
            descriptors = [
                calculate_properties(smi, descriptors=DEFAULT_DESCRIPTORS)
                for smi in smiles
            ]
            X = np.stack(descriptors)

        if self.scaler is None:
            self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.model.fit(X, targets)
        return self

    def predict(self, X: str | list[str] | np.ndarray) -> np.ndarray:
        assert self.scaler is not None, "Scaler has not been fitted."
        if not isinstance(X, np.ndarray):
            if isinstance(X, str):
                X = [X]
            smiles = [canonicalize_smiles(smi) for smi in X]
            descriptors = [
                calculate_properties(smi, descriptors=DEFAULT_DESCRIPTORS)
                for smi in smiles
            ]
            X = np.stack(descriptors)

        X = self.scaler.transform(X)

        predictions = self.model.predict(X)
        return predictions

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_handle:
            pickle.dump((self.model, self.scaler), file_handle)

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as file_handle:
            model, scaler = pickle.load(file_handle)
        return cls(model, scaler)


class EnsemblePermeabilitySurrogate:
    def __init__(self, models: list[PermeabilitySurrogate]):
        self.models = models

    def predict(self, X: str | list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray):
            if isinstance(X, str):
                X = [X]
            smiles = [canonicalize_smiles(smi) for smi in X]
            descriptors = [calculate_properties(smi) for smi in smiles]
            X = np.stack(descriptors)

        predictions = np.array([model.predict(X) for model in self.models])
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        return mean_predictions, std_predictions

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            model_name = f"model_{self.models.index(model)}.pkl"
            model_path = path / model_name
            model.save(model_path)

    @classmethod
    def from_hf(cls, hf_data: str, model_path: str = "ensemble"):
        local_dir = snapshot_download(repo_id=hf_data)
        return cls.from_directory(local_dir + f"/{model_path}")

    @classmethod
    def from_directory(cls, path: str | Path):
        path = Path(path)
        models = []
        for model_path in path.glob("*.pkl"):
            models.append(PermeabilitySurrogate.from_file(model_path))
        return cls(models)
