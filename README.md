# The Challenges of PROTAC Permeability Prediction

### Code Structure
```sh
protac_permeability
├── chem_utils.py                       # utils for parsing and training
├── data_mining                         
│   ├── config          
│   │   └── llm_extraction_config.yaml  # config for extern/LLM-TPD-Extraction LLM extraction
│   ├── miner.py                        # DOI based publication miner
│   ├── parse_protacs.py                # script for parsing new raw data
│   └── prompts
│       └── prompt_pampa.md             # prompt for extern/LLM-TPD-Extraction LLM extraction
├── dataset
│   └── make_dataset.py                 # script for combining PROTAC-DB 3.0 data with the newly mined data
├── permeability_surrogate
│   ├── __init__.py
│   ├── surrogate_model.py              # model definition
│   └── fit_surrogate.py                # model training
├── paper_figures.ipynb                 # paper figures notebook
└── plot_style.py
```

### Data Structure
```sh
data
├── dois.csv                            # mined publication dois
├── new_protacs_raw.csv                 # new raw data 
├── new_protacs_parsed.csv              # parsed new data with unified units and canonical SMILES
├── protacdb.csv                        # PROTAC-DB 3.0 dataset
├── protacdb_filtered.csv               # filtered records from PROTAC-DB 3.0 with PAMPA measurements
└── combined_protacs.csv                # final dataset
```

### Dependencies
The dependencies in this project are managed by `uv`, to install them run 
```sh
uv sync
```
For mining or plotting dependencies use
```sh
uv sync --extra mining 
```
or
```sh
uv sync --extra plotting
```

###  Pipeline
#### 1) LLM Data Mining
The mining step is done using an external repository in `extern/LLM-TPD-Extraction`. For this it is necessary to create an `.env` file as described in `extern/LLM-TPD-Extraction/README.md`. Note that, since this is an external fork, it cannot be anonymized and is therefore excluded from this repository for the purposes of the review.

First mining the publications automatically can be done as

```sh
uv run protac_permeability/data_mining/miner.py \
    --doi_csv data/dois.csv
    --out_dir data
```
Then, it is possible to run the data mining code as
```sh
cd extern/LLM-TPD-Extraction
pixi run python scripts/run_pipeline.py \
    --config_path ../../protac_permeability/data_mining/config/llm_extraction_config.yaml \
    --input_prompt ../../protac_permeability/data_mining/prompts/prompt_pampa.md \
    --cleaning \
    --llm \
    --with_history \
    --use_api \
    --csv
```

This extracts data from the listed publications and outputs a combined csv file with all mined data points. However, most data points do not have associated SMILES strings / IUPAC names, so it is necessary to go through the sucessfully mined papers to verify that the mined data is correct, the mined sructures are actually PROTACs, and to add any missing entries, SMILES or IUPAC names.

#### 2) Dataset Creation

To create the dataset from a csv of unparsed entries, run

```sh
uv run protac_permeability/data_mining/parse_protacs.py \
    --raw_csv data/new_protacs_raw.csv \
    --parsed_csv data/new_protacs_parsed.csv
```

which outputs a parsed csv with unified PAMPA units, IUPAC names transformed to SMILES strings and canonicalized SMILES. Then this parsed dataset can be combined with PROTAC-DB 3.0 using

```sh
uv run protac_permeability/dataset/make_dataset.py \
    --extracted_protacs_path ./data/new_protacs_parsed.csv \
    --out_path data/combined_protacs.csv
```
which returns the final dataset.

#### 3) Model Training
To train the models on the new data run
```sh
uv run protac_permeability/permeability_surrogate/fit_surrogate.py \
    --data_path data/combined_protacs.csv \
    --save_dir models/test_model \
    --descriptor_cols MolecularWeight CharVol cLogP HeavyAtomCount RingCount HydrogenBondAcceptorCount HydrogenBondDonorCount RotatableBondCount TopologicalPolarSurfaceArea FractionCSP3 NumStereoCenters AllBonds RingAtoms Halogens HeteroAtoms TNSA Flexibility
```
For optional arguments:
```sh
uv run protac_permeability/permeability_surrogate/fit_surrogate.py --help
```

### Figures
To reproduce figures and results from the paper, run `protac_permeability/paper_figures.ipynb`. You need to install `plotting` dependencies.

### AI Usage Statement
During the development of the code we used LLM tools like Claude and Gemini for helping with coding, code refactoring and writing documentation.
