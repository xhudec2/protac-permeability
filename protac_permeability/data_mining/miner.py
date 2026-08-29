# Developed by Yaochen Rao https://github.com/yaochenr/PMC_Data_Mining
import json
from argparse import ArgumentParser
from pathlib import Path

from pmc_miner import DOIBasedMiner


def mine_doi_papers(doi_csv: str, out_dir: str) -> None:
    doi_miner = DOIBasedMiner(
        output_dir=out_dir,
        # Difficult to map images to the correct location in the paper, so they are not downloaded by default
        download_images=False,
    )

    _ = doi_miner.mine_from_csv(doi_csv)

    mined_dois = []
    out_dir_path = Path(out_dir)
    for dir in out_dir_path.iterdir():
        if not dir.is_dir() or dir.stem == "logs":
            continue
        mined_dois.append({"pmc_id": dir.stem})

    json.dump(mined_dois, open(out_dir_path / "papers.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Mine papers from a CSV of DOIs")
    parser.add_argument(
        "--doi_csv",
        type=Path,
        required=True,
        help="Path to the CSV file containing DOIs to mine.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Path to the directory where mined papers will be saved.",
    )
    args = parser.parse_args()
    mine_doi_papers(args.doi_csv, args.out_dir)
