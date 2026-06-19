# PAMPA Extraction Prompt

## Goal

You will be given a single scientific paper. The paper is delimited by triple backticks (``` ... ```). Your goal is to extract **PAMPA** (Parallel Artificial Membrane Permeability Assay) permeability measurements, specifically focusing on **apparent permeability** values reported as **Papp** (and closely synonymous PAMPA permeability metrics like Pe when explicitly described as PAMPA).

## Required Fields

Extract the following 10 fields:

- Compound_Name: Compound identifier/name exactly as reported (e.g., "1a", "Compound 7", "AZD1234"); normalize whitespace/case but preserve suffixes (a/b) and punctuation that distinguishes compounds.
- IUPAC_Name: IUPAC name if explicitly provided; otherwise leave blank.
- SMILES: SMILES if explicitly provided; otherwise leave blank.
- Assay: The assay name as reported (should be "PAMPA" or a paper-specific variant like "PAMPA-BBB").
- Permeability_Metric: The metric label as reported (e.g., "Papp", "Pe", "logPapp", "logPe"). Only populate when the paper ties the metric to PAMPA.
- Papp: The reported permeability value string exactly as shown (preserve qualifiers like ~, <, >, ranges, "+/-", and scientific notation).
- Papp_units: The units exactly as reported (e.g., "cm/s", "10^-6 cm/s", "×10−6 cm/s"). If units are only in a column header, still populate from that header.
- Figure: The figure/scheme/table identifier(s) where the **compound name and/or structure depiction** is defined (i.e., where you would find the molecule image for that compound) when the paper uses placeholders like `<FIGURE fig...>`, `<SCHEME sch...>`, or `<TABLE tab...>` (e.g., "fig2", "sch1", "tab3"). If multiple identifiers apply, include **all** of them as a single string separated by `;` then a space (example: "fig2; sch1"); otherwise leave blank.
- Notes: Any short, value-critical context that cannot be represented in the fields above (e.g., "PAMPA-BBB", "artificial membrane: hexadecane", "stirring", "sink conditions"). Keep this concise; do not speculate.
- `Figure` is for compound/structure provenance (mapping the molecule image to the compound), not for the PAMPA value provenance.

## Extraction Principles

- Only extract permeability for **PROTAC compounds** (heterobifunctional degraders). Do **not** emit records for non-PROTAC compounds such as warheads/inhibitors, E3 ligase ligands (e.g., VHL/CRBN binders), linkers, fragments, metabolites, or other monovalent controls.
- A compound counts as a PROTAC only when the paper explicitly describes it as a PROTAC/degrader/heterobifunctional/bifunctional molecule (or clearly shows it as a warhead–linker–E3-ligand construct in structures/schemes). If uncertain whether a compound is a PROTAC, **do not** emit a record.
- Only extract values that are explicitly described as PAMPA measurements. Do **not** include Caco-2, MDCK, RRCK, or in vivo permeability values.
- Do not infer or calculate: do not convert between log and linear scale; do not convert units; do not compute Papp from % transported.
- Preserve reported qualifiers exactly (e.g., "<0.1", ">50", "~3"). Do not round or coerce numeric strings.
- The paper text may contain placeholders that stand in for rendered assets, such as `<FIGURE fig1...>`, `<SCHEME sch2...>`, `<TABLE tab3...>`, or similar.
- Populate `Figure` from the placeholder(s) where the compound is defined/shown (structure image, scheme, or defining table), even if the PAMPA value is reported elsewhere.
- If the compound name/structure appears in multiple placeholders across the paper, include all unique identifiers in `Figure` (joined with `;` then a space, in order of first appearance).
- Table discipline: when parsing tables, ensure each Papp/Pe value is associated with the correct compound row and the correct experimental condition column (assay variant, membrane/solvent system, etc. when present).
- If a table reports multiple PAMPA variants (e.g., PAMPA-GI vs PAMPA-BBB), create separate objects per distinct assay/condition.
- If the paper reports only a permeability classification (e.g., "high/low") with no numeric Papp/Pe value, do **not** emit a record.

## Output Format

- Output a single JSON array of objects, with no added commentary or explanation.
- Each object corresponds to a single PAMPA permeability measurement for one compound under one condition (paper + assay variant and any key condition notes if given).
- Each object must contain **only** the fields listed under 'Required Fields'.

Example: [{"Compound_Name": "7a", "IUPAC_Name": "", "SMILES": "", "Assay": "PAMPA", "Permeability_Metric": "Papp", "Papp": "2.3", "Papp_units": "10^-6 cm/s", "Figure": "fig2; sch1", "Notes": "PAMPA-BBB"}]
