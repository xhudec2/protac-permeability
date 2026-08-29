import argparse

import numpy as np
import pandas as pd

from protac_permeability.chem_utils import (
    calculate_properties,
    canonicalize_smiles,
    descriptor_functions,
    try_float,
)


def parse_protacdb(protacdb_path: str) -> pd.DataFrame:
    protac_db = pd.read_csv(protacdb_path)
    filtered_protac_db = protac_db[~protac_db["PAMPA Papp (nm/s, Permeability)"].isna()]
    filtered_protac_db = filtered_protac_db[
        [
            "Compound ID",
            "Smiles",
            "PAMPA Papp (nm/s, Permeability)",
            "Article DOI",
            "Target",
            "E3 ligase",
        ]
    ]
    filtered_protac_db.rename(
        columns={
            "PAMPA Papp (nm/s, Permeability)": "PAMPA",
            "Compound ID": "protac_db_id",
            "Smiles": "SMILES",
            "E3 ligase": "E3_ligase",
        },
        inplace=True,
    )
    filtered_protac_db["PAMPA"] = filtered_protac_db["PAMPA"].apply(try_float)
    filtered_protac_db.dropna(subset=["PAMPA"], inplace=True)
    # PROTAC-DB has a unit error in the PAMPA values for all articles apart
    # from 10.1021/acs.jmedchem.8b01413, which is corrected here
    filtered_protac_db["PAMPA"] = filtered_protac_db["PAMPA"] * np.where(
        [
            "10.1021/acs.jmedchem.8b01413" in x
            for x in filtered_protac_db["Article DOI"]
        ],
        1,
        10,
    )
    filtered_protac_db["logPAMPA"] = np.log10(filtered_protac_db["PAMPA"]) - 7
    filtered_protac_db["SMILES"] = filtered_protac_db.SMILES.apply(canonicalize_smiles)
    filtered_path = protacdb_path.replace(".csv", "_filtered.csv")
    filtered_protac_db.to_csv(filtered_path, index=False)
    return filtered_protac_db


def parse_extracted_protacs(extracted_protacs_path: str) -> pd.DataFrame:
    new_protacs = pd.read_csv(extracted_protacs_path)
    new_protacs = new_protacs.rename(
        columns={
            "smiles": "SMILES",
            "doi": "Article DOI",
            "pampa": "PAMPA",
            "e3": "E3_ligase",
            "poi": "Target",
        }
    )
    new_protacs["logPAMPA"] = np.log10(new_protacs["PAMPA"]) - 7
    new_protacs = new_protacs.drop(columns=["compound_name", "pampa_unit"])
    new_protacs["SMILES"] = new_protacs["SMILES"].apply(canonicalize_smiles)
    return new_protacs


def join_dois(dois: pd.Series) -> str:
    if dois.nunique() == 1:
        return dois.iloc[0]

    doi_str = ""
    for doi in dois.dropna().unique():
        if doi_str != "":
            doi_str += ";"
        doi_str += doi.replace(" ", "")
    return doi_str


def join_datasets(
    protacdb_path: str, extracted_protacs_path: str, out_path: str
) -> None:
    protacdb_df = parse_protacdb(protacdb_path)
    extracted_df = parse_extracted_protacs(extracted_protacs_path)

    combined_df = pd.concat([extracted_df, protacdb_df], ignore_index=True)
    combined_df["SMILES"] = combined_df["SMILES"].apply(canonicalize_smiles)
    combined_df = combined_df.dropna(subset=["SMILES"])

    # collect all DOIs for a given SMILES as "doi1;doi2;..." before deduplicating
    combined_df["Article DOI"] = combined_df.groupby("SMILES")["Article DOI"].transform(
        join_dois
    )
    # for duplicated SMILES: keep the first row if PAMPA values disagree (newly mined data comes first),
    # and if they agree, keep the last row (PROTAC-DB come first)
    pampa_nunique = combined_df.groupby("SMILES")["PAMPA"].transform("nunique")
    disagreeing_pampa = pampa_nunique > 1
    combined_df = pd.concat(
        [
            combined_df[disagreeing_pampa].drop_duplicates(
                subset=["SMILES"], keep="first"
            ),
            combined_df[~disagreeing_pampa].drop_duplicates(
                subset=["SMILES"], keep="last"
            ),
        ]
    ).sort_index()
    combined_df = combined_df.reset_index()
    combined_df = combined_df.rename(columns={"index": "protac_id"})

    protacs_features_2d = pd.DataFrame(
        combined_df["SMILES"].apply(calculate_properties).values.tolist()
    )
    protacs_features_2d["protac_id"] = combined_df.protac_id
    protacs_merged = combined_df.merge(
        protacs_features_2d, on=["protac_id"], how="outer"
    )
    protacs_merged = protacs_merged.drop(columns=["protac_id"])
    protacs_merged = protacs_merged.reset_index()
    protacs_merged = protacs_merged.rename(columns={"index": "protac_id"})
    protacs_merged.columns = list(protacs_merged.columns[:-17]) + list(
        descriptor_functions.keys()
    )
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
        default="./data/new_protacs_parsed.csv",
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
