import argparse

import numpy as np
import pandas as pd

from protac_permeability.chem_utils import (
    calculate_properties,
    canonicalize_smiles,
    try_float,
)


def parse_protacdb(protacdb_path: str) -> pd.DataFrame:
    protac_db = pd.read_csv(protacdb_path)
    filtered_protac_db = protac_db[~protac_db["PAMPA Papp (nm/s, Permeability)"].isna()]
    filtered_protac_db = filtered_protac_db[
        ["Compound ID", "Smiles", "PAMPA Papp (nm/s, Permeability)", "Article DOI"]
    ]
    filtered_protac_db.rename(
        columns={
            "PAMPA Papp (nm/s, Permeability)": "PAMPA_Papp",
            "Compound ID": "protac_id",
            "Smiles": "SMILES",
        },
        inplace=True,
    )
    filtered_protac_db["PAMPA_Papp"] = filtered_protac_db["PAMPA_Papp"].apply(try_float)

    # PROTACDB has a unit error in the Papp values for all articles apart from 10.1021/acs.jmedchem.8b01413, which is corrected here
    filtered_protac_db["PAMPA_Papp"] = filtered_protac_db["PAMPA_Papp"] * np.where(
        [
            "10.1021/acs.jmedchem.8b01413" in x
            for x in filtered_protac_db["Article DOI"]
        ],
        1,
        10,
    )
    filtered_protac_db["PAMPA"] = np.log10(filtered_protac_db["PAMPA_Papp"]) - 7
    filtered_protac_db["SMILES"] = filtered_protac_db.SMILES.apply(canonicalize_smiles)
    return filtered_protac_db


def parse_extracted_protacs(extracted_protacs_path: str) -> pd.DataFrame:
    new_protacs = pd.read_csv(extracted_protacs_path)
    new_protacs = new_protacs.reset_index()
    new_protacs = new_protacs.rename(
        columns={
            "index": "protac_id",
            "smiles": "SMILES",
            "doi": "Article DOI",
            "pampa": "PAMPA_Papp",
        }
    )
    new_protacs["protac_id"] = (new_protacs["protac_id"] + 1) * 1000000
    new_protacs["PAMPA"] = np.log10(new_protacs["PAMPA_Papp"]) - 7
    new_protacs = new_protacs.drop(columns=["compound_name", "pampa_unit"])
    new_protacs["SMILES"] = new_protacs["SMILES"].apply(canonicalize_smiles)
    return new_protacs


def join_datasets(
    protacdb_path: str, extracted_protacs_path: str, out_path: str
) -> None:
    protacdb_df = parse_protacdb(protacdb_path)
    extracted_df = parse_extracted_protacs(extracted_protacs_path)

    combined_df = pd.concat([extracted_df, protacdb_df], ignore_index=True)
    combined_df["SMILES"] = combined_df["SMILES"].apply(canonicalize_smiles)
    # keep the newly extracted PROTACs if duplicates exist
    combined_df = combined_df.drop_duplicates(subset=["SMILES"], keep="first")
    combined_df = combined_df.dropna(subset=["SMILES"])
    combined_df = combined_df.reset_index(drop=True)

    protacs_features_2d = pd.DataFrame(
        combined_df["SMILES"].apply(calculate_properties).values.tolist()
    )
    protacs_features_2d["SMILES"] = combined_df["SMILES"]
    protacs_features_2d["protac_id"] = combined_df.protac_id
    protacs_merged = combined_df.merge(
        protacs_features_2d, on=["protac_id", "SMILES"], how="outer"
    )
    protacs_merged = protacs_merged.dropna().reset_index(drop=True)
    protacs_merged.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join the PROTACDB dataset with the newly extracted PROTACs dataset"
    )
    parser.add_argument(
        "--protacdb_path",
        type=str,
        default="./data/protacdb.csv",
        help="Path to the PROTACDB dataset CSV file",
    )
    parser.add_argument(
        "--extracted_protacs_path",
        type=str,
        default="./data/extracted_protacs.csv",
        help="Path to the newly extracted PROTACs dataset CSV file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/combined_protacs.csv",
        help="Path to the output combined dataset CSV file",
    )
    args = parser.parse_args()

    join_datasets(
        protacdb_path=args.protacdb_path,
        extracted_protacs_path=args.extracted_protacs_path,
        out_path=args.out_path,
    )
