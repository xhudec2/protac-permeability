from argparse import ArgumentParser

import numpy as np
import pandas as pd
from py2opsin import py2opsin

from protac_permeability.chem_utils import canonicalize_smiles


def to_smiles(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    return py2opsin(
        chemical_name=x,
        output_format="SMILES",
    )


def parse_pampa(pampa: str, unit: str) -> float:
    if isinstance(pampa, str) and (pampa[0] == "<" or pampa[0] == ">"):
        pampa = pampa[1:]
    pampa_val = float(pampa)
    match unit:
        case "10e-6 cm/s":
            pampa_val = pampa_val * 10
        case "-log (10-6 cm/s)":
            pampa_val = np.power(10, -pampa_val + 7)
        case "log (10-6 cm/s)":
            pampa_val = np.power(10, pampa_val + 7)
    return pampa_val


def parse_protacs(raw_csv: str, parsed_csv: str) -> None:
    df = pd.read_csv(raw_csv)
    df["smiles"] = np.where(df.smiles.isna(), df.iupac.apply(to_smiles), df.smiles)
    df["pampa"] = df.apply(
        lambda row: parse_pampa(row.pampa, row.pampa_unit), axis=1
    ).astype(np.float32)
    df["pampa"] = np.clip(df["pampa"], a_min=1e-3, a_max=np.inf)
    df["pampa_unit"] = "10e-6 nm/s"
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
