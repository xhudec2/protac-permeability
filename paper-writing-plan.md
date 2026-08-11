# Paper Writing Plan — PROTAC Permeability

*Last updated: August 10th, 2026.*

This document has two parts: (1) a factual summary of what this repo currently
does, derived from reading the code and data, and (2) a plan for turning that
into a manuscript using the `templates/latex/main.tex` skeleton. Numbers
marked **[TBD]** need to be pulled from an actual run — they are not yet
recorded anywhere in the repo.

---

## 1. What the repo actually does

**Core question:** can passive membrane permeability (PAMPA Papp) of PROTACs
be predicted from 2D structure, given that PROTACs are large,
beyond-rule-of-5 heterobifunctional molecules (warhead–linker–E3 ligand) for
which permeability is a well-known bottleneck and public data is sparse?

### 1.1 Data mining pipeline (`protac_permeability/data_mining/`)
- `miner.py` pulls full-text open-access papers from PubMed Central for a
  list of DOIs (`data/dois.csv`) using the external `pmc-miner` package
  (optional dependency, `DOIBasedMiner`). Output: ~140 paper folders under
  `data/mined_dois/` plus a `papers.json` catalog.
- Extraction of PAMPA records from the mined full text is delegated to
  `extern/LLM-TPD-Extraction` — a vendored sibling project (own paper:
  *"Beyond Manual Curation: Augmenting Targeted Protein Degradation Databases
  via Agentic Literature Extraction Workflows"*) that runs an LLM (via
  OpenRouter, e.g. GPT-5) over cleaned paper text with a prompt-refinement
  loop (CAPO). It is excluded from the installable package
  (`exclude = ["extern*"]`) — used as a tool, not shipped.
- `data_mining/prompts/prompt_pampa.md` is this project's task-specific
  prompt: extract only PAMPA/Pe values, only for compounds explicitly
  described as PROTACs/degraders (not warheads, E3 ligands, linkers,
  metabolites), preserving qualifiers (`<`, `>`, `~`) and recording
  figure/table provenance for the compound structure.
- `data_mining/config/llm_extraction_config.yaml` wires this together
  (input/output dirs, fields to strip, results dir
  `data/mined_protacs/`).
- **Note:** the config file has hardcoded absolute local paths
  (`/Users/xhudec2/Documents/aime/...`) — will need to be relativized (per
  the conventions guide) before this is reproducible by anyone else, and
  before it's cited as a method in a paper.

### 1.2 Parsing and dataset construction (`data_mining/parse_protacs.py`, `dataset/`)
- `parse_protacs.py`: takes the raw LLM-extracted CSV, converts IUPAC names
  to SMILES via OPSIN (`py2opsin`) when SMILES is missing, normalizes PAMPA
  units to a common scale, clips values at a floor, canonicalizes SMILES
  (RDKit, largest-fragment/salt stripping), dedups. Result: `data/new_protacs.csv`
  (**62 rows**, i.e. ~61 compounds — small).
- `dataset/make_dataset.py`: joins two sources —
  - **PROTAC-DB** (`data/protacdb.csv`, 9,381 rows total) filtered to rows
    with a non-null PAMPA Papp value. Applies a manual ×10 unit correction to
    all articles *except* DOI `10.1021/acs.jmedchem.8b01413`, which the code
    comments describe as fixing "a unit error in the Papp values ... for all
    articles apart from" that one DOI — this is a real, non-obvious data
    quality finding worth stating explicitly in the paper's data section.
  - **Newly LLM-extracted PROTACs** (`data/new_protacs.csv`).
  - Both sides: `PAMPA = log10(Papp) - 7` transform, SMILES canonicalized,
    deduplicated on SMILES (new extractions win over PROTAC-DB on conflict),
    then 18 2D descriptors computed and merged.
  - Final output: `data/protacs_merged.csv` — **88 compounds**. This is the
    actual training set size for the surrogate model. Worth flagging early:
    this is a small-data regime for ML, which should shape how the methods
    and limitations sections are framed.

### 1.3 Featurization (`chem_utils.py`)
- `canonicalize_smiles`: RDKit canonical SMILES + `LargestFragmentChooser`.
- `calculate_properties`: 18 descriptors — MW, exact mass, cLogP, heavy atom
  count, ring count, HBA/HBD, rotatable bonds, TPSA, fraction Csp3,
  stereocenter count, bond count, ring-atom count, halogen count, heteroatom
  count, a custom **TNSA** ("total non-polar surface area" =
  20 × carbon count − TPSA, floored at 0), a `SizeShape` term (ring-atom
  count, duplicated with a different name), and a **flexibility** index
  (rotatable bonds / total bonds). This descriptor set (TNSA, size/shape,
  flexibility) mirrors the kind of "beyond Ro5" descriptor sets used in
  published PROTAC/macrocycle permeability QSAR work — worth identifying and
  citing the specific source paper this was adapted from, if there is one.

### 1.4 Surrogate model (`permeability_surrogate/`)
- `PermeabilitySurrogate`: sklearn regressor + `StandardScaler` wrapper;
  fits/predicts directly from SMILES (auto-featurizes) or precomputed arrays;
  picklable via `save`/`from_file`.
- `EnsemblePermeabilitySurrogate`: bag of the above; `predict` returns
  **(mean, std)** across models, i.e. gives an uncertainty estimate for free.
  Can load a saved ensemble from a local directory or from the Hugging Face
  Hub (`ailab-bio/permeability-surrogate`, via `snapshot_download`) — so the
  model is already set up for public release.
- `fit_surrogate.py`: trains a **Ridge regression** ensemble on the 18
  descriptors, target = `log10(PAMPA) - 7`. For `num_repeats` (default 5) ×
  `n_models`-fold (default 5) stratified CV (stratified on quantile-binned
  target, `RANDOM_SEED = 42`), each outer fold does an inner 2-fold CV +
  Optuna TPE search (100 trials) to pick Ridge `alpha` maximizing Spearman
  correlation, refits on the full outer-train split, and reports held-out
  Spearman correlation. Default: **25 models** in the ensemble
  (5 repeats × 5 folds). `models/ensemble/` (current) and
  `models/ensemble_old/` (prior version) are both checked into the repo.
  Commit `0d3a674 fix seed for reproducibility` indicates the team already
  cares about this — good, matches the reproducibility checklist in
  `templates/`.
- Held-out Spearman correlation, printed by `fit_surrogate.py` but not
  currently saved anywhere in the repo: **[TBD — rerun and record]**.

### 1.5 3D / conformational analysis (`dataset/make_compact.py`, untracked)
- Separate from the 2D-descriptor pipeline above. Operates on QM-optimized
  conformer ensembles (SDF files) per PROTAC and, using RDKit + `mdtraj`,
  computes intramolecular H-bonds, the ring size of the largest
  H-bond-closed macrocycle, radius of gyration, 3D PSA, and SASA — then picks
  the conformer with the largest intramolecular H-bond ring (i.e. the most
  "compact"/chameleonic pose) and exports it as a PDB.
- This targets a genuinely different, well-known hypothesis for PROTAC
  permeability: that *conformational flexibility / intramolecular H-bonding
  ("molecular chameleonicity")* — not just static 2D descriptors — governs
  passive permeability of large flexible molecules.
- **Status: not yet wired into the feature set or the surrogate model**, and
  the file is untracked in git. This is either (a) exploratory/future work,
  or (b) an intended second modeling axis for the paper. This needs a
  decision before writing the methods section — see open questions below.

### 1.6 What `templates/` is
Not part of the software pipeline — it's the AIME lab's shared
paper-writing kit: `aime.cls` + `main.tex` skeleton, a matplotlib style
guide + `aime_style.py`, and a conventions guide. This is what the actual
manuscript will be written in.

---

## 2. Open questions to resolve before/while writing

1. **Is the 3D/conformer analysis (`make_compact.py`) going into this paper?**
   If yes, it needs its own subsection (data: QM-optimized SDF source and
   level of theory; method: H-bond/ring-size/Rg/3D-PSA definitions) and
   probably its own results (does compactness correlate with measured PAMPA
   better than 2D TNSA/flexibility?). If no, leave out of methods and
   mention only as future work.
2. **What is the actual held-out performance number?** Need to rerun
   `fit_surrogate.py` on `data/protacs_merged.csv` (or the current canonical
   dataset) and record mean ± std Spearman correlation for the results
   section/table.
3. **Ablation the paper needs:** does adding the LLM-mined 61 compounds
   improve on PROTAC-DB-only training? This is the direct evidence for the
   "LLM-assisted mining helps" contribution claim and should be a table row
   (PROTAC-DB only vs. PROTAC-DB + mined).
4. **Extraction pipeline quality:** does `LLM-TPD-Extraction` (or this
   project specifically) report precision/recall of the PAMPA extraction
   against a ground-truth/manually-checked subset? If so, cite/reuse those
   numbers in the data section; if not, a small manual spot-check of the 61
   mined records would substantially strengthen the data-quality claim.
5. **n = 88 is small.** Decide up front how the paper frames this: as a
   proof-of-concept/pilot data-mining contribution rather than a
   state-of-the-art permeability predictor, since the interesting/novel
   claim is really the data pipeline + uncertainty-aware ensemble, not
   beating some benchmark.
6. **Source of the descriptor set.** `calculate_properties` looks adapted
   from a specific published PROTAC-permeability descriptor scheme (TNSA,
   size/shape, flexibility terms) — track down and cite that source rather
   than presenting it as novel.
7. **Hardcoded local paths** in `llm_extraction_config.yaml` should be fixed
   to relative paths before claiming reproducibility in the paper (per the
   conventions guide's own checklist).

---

## 3. Section-by-section plan for `main.tex`

- **Abstract** — Context: PROTAC permeability is a major developability
  bottleneck for an otherwise powerful modality; beyond-Ro5 chemical space
  breaks standard permeability heuristics. Gap: public PAMPA data for
  PROTACs is sparse and concentrated in very few sources (only ~9,380/…  of
  PROTAC-DB rows even have a PAMPA value, effectively from one paper's
  assay). Approach: an LLM-assisted literature-mining pipeline that adds new
  PAMPA records beyond PROTAC-DB, plus a descriptor-based ensemble
  regression surrogate with built-in uncertainty. Results: dataset grows
  from PROTAC-DB-only to 88 compounds combined; ensemble achieves Spearman
  ρ = **[TBD]**; released as an open, pip-installable model
  (`ailab-bio/permeability-surrogate` on HF Hub). Significance: a
  reproducible, extensible recipe for growing small permeability datasets
  and getting calibrated uncertainty from them, applicable beyond PAMPA/PROTACs.

- **Introduction**
  1. Motivation: PROTACs' therapeutic promise vs. their permeability
     liability; why passive PAMPA permeability specifically matters (as a
     proxy for oral/cell-based bioavailability, contrasted with active
     transport-dependent Caco-2/MDCK).
  2. Prior work: PROTAC-DB and other curated resources; existing
     descriptor/ML models for beyond-Ro5 permeability; LLM-based literature
     extraction for chemistry (cite `extern/LLM-TPD-Extraction`'s own paper).
  3. Gap: PAMPA coverage in PROTAC-DB is small, concentrated in one source
     series, and (per this repo's own finding) had a unit-consistency issue
     silently affecting most rows — small, noisy, low-diversity data limits
     any predictive model.
  4. Contributions: (i) LLM-assisted extraction pipeline with a
     PROTAC/PAMPA-specific prompt that adds new records beyond PROTAC-DB,
     (ii) a cleaned, combined, versioned dataset, (iii) an uncertainty-aware
     ensemble surrogate model releasable via Hugging Face Hub, (iv)
     [if kept] a conformational-compactness analysis as a complementary
     structural determinant.

- **Background** — PAMPA assay basics; PROTAC-DB; brief survey of prior
  permeability QSAR descriptor sets for large/flexible molecules (source of
  TNSA/flexibility terms); one paragraph on LLM literature extraction
  methodology (CAPO / agentic extraction), citing the sibling paper rather
  than re-deriving it.

- **Methods → Data** — DOI list → PMC mining → LLM extraction (prompt +
  config, PROTAC-only qualifier-preserving extraction rules) → parsing/unit
  normalization (OPSIN name-to-structure fallback) → merge with PROTAC-DB
  (state the unit-correction explicitly, and the "keep new on duplicate"
  policy) → final dataset stats table (n, PAMPA range/distribution, source
  breakdown PROTAC-DB vs. mined).

- **Methods → Featurization** — the 18 descriptors, grouped by category
  (size, polarity/H-bonding, shape/flexibility, stereochemistry); define
  TNSA and the flexibility index explicitly since they're non-standard.

- **Methods → Model** — Ridge regression, ensembling scheme (5×5 repeated
  stratified CV), nested Optuna hyperparameter search (search space, 100
  trials, objective = Spearman on inner holdout), seed. State this in terms
  the reproducibility checklist expects: optimizer/model, search strategy,
  seeds, compute (this is cheap/CPU-only, worth noting).

- **Methods → [optional] Conformational analysis** — only if question 1
  above is resolved "yes": QM conformer generation method/level of theory
  (need to find where the SDFs came from — not in this repo currently),
  H-bond/ring-size/Rg/3D-PSA definitions from `make_compact.py`.

- **Results** — key figure: predicted vs. observed PAMPA (ensemble mean ±
  std as error bars) on held-out folds. Table: Spearman ρ (and
  RMSE/R² if computed) mean ± std across repeats, PROTAC-DB-only vs.
  +mined-data ablation (open question 3). If kept: correlation between
  conformational compactness metrics and measured/predicted PAMPA.

- **Discussion** — interpret why descriptor-ensemble + Ridge is a reasonable
  choice at n = 88 (linear model, regularized, avoids overfitting vs. a
  larger/nonlinear model); what the LLM-mined data added qualitatively
  (chemotype diversity vs. just more points from the same series);
  relationship (or lack of) to conformational flexibility if that section is
  included.
  - **Limitations:** small n (88); single assay type (PAMPA only, no
    cellular/BBB permeability); extraction pipeline precision not yet
    quantified on this specific task (unless resolved per open question 4);
    linear model may not capture nonlinear structure-permeability
    relationships; local-path reproducibility issue in the mining config.

- **Conclusion** — recap: built an LLM-assisted mining pipeline to grow a
  PROTAC PAMPA dataset, trained an uncertainty-aware ensemble surrogate,
  released model + (eventually) data publicly.

- **Software and data availability** — GitHub repo (this one — set the URL
  in `main.tex`'s `\codeavailability`), HF Hub model repo already exists
  (`ailab-bio/permeability-surrogate`), data should go to Zenodo per the
  conventions guide once finalized (`\dataavailability` still has the
  placeholder DOI).

---

## 4. Before drafting: concrete TODOs

- [ ] Rerun `fit_surrogate.py` on the current `data/protacs_merged.csv`,
      record the reported Spearman mean ± std.
- [ ] Run the PROTAC-DB-only vs. +mined ablation (same script, two data files).
- [ ] Decide on `make_compact.py` / conformational analysis: in scope or future work.
- [ ] Track down citation for the 18-descriptor set (TNSA/flexibility/size-shape).
- [ ] Relativize paths in `llm_extraction_config.yaml`.
- [ ] Fill in `\codeavailability`/`\dataavailability` URLs in `templates/latex/main.tex`.
- [ ] Decide fate of `test.py` and `make_compact.py` (untracked) — commit,
      move under a `scripts/`-type folder, or drop before release.
