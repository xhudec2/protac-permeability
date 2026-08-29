from argparse import ArgumentParser

import numpy as np
import pandas as pd
from py2opsin import py2opsin

from protac_permeability.chem_utils import canonicalize_smiles, parse_pampa


def to_smiles(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    return py2opsin(
        chemical_name=x,
        output_format="SMILES",
    )


def parse_protacs(raw_csv: str, parsed_csv: str) -> None:
    df = pd.read_csv(raw_csv)
    df["smiles"] = np.where(df.smiles.isna(), df.iupac.apply(to_smiles), df.smiles)
    df["original_pampa"] = df["pampa"]
    df["pampa"] = df.apply(
        lambda row: parse_pampa(row.pampa, row.pampa_unit), axis=1
    ).astype(np.float32)
    df.dropna(subset=["smiles", "pampa"], inplace=True)
    df["pampa"] = np.clip(df["pampa"], a_min=1e-3, a_max=np.inf)
    df["pampa_unit"] = "nm/s"
    df = df.drop(columns=["iupac"])
    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    df = df.drop_duplicates(subset=["smiles"], keep="first")
    df.to_csv(parsed_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Parse raw PROTACs CSV to a more usable format")
    parser.add_argument(
        "--raw_csv",
        type=str,
        required=True,
        help="Path to the raw PROTACs CSV file.",
    )
    parser.add_argument(
        "--parsed_csv",
        type=str,
        required=True,
        help="Path to the parsed PROTACs CSV file.",
    )

    args = parser.parse_args()
    parse_protacs(
        raw_csv=args.raw_csv,
        parsed_csv=args.parsed_csv,
    )
