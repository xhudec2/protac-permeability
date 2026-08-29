import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina


def canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    lfg = rdMolStandardize.LargestFragmentChooser()
    mol = lfg.choose(mol)
    return Chem.MolToSmiles(mol)


def try_float(pampa):
    try:
        if isinstance(pampa, str):
            pampa = pampa.strip()
            if pampa[0] == "<" or pampa[0] == ">":
                pampa = pampa[1:]
        pampa_val = float(pampa)
        return pampa_val
    except Exception as _:
        return np.nan


def parse_pampa(pampa: str, unit: str) -> float:
    pampa_val = try_float(pampa)
    if np.isnan(pampa_val):
        return np.nan
    match unit:
        case "10e-6 cm/s":
            pampa_val = pampa_val * 10
        case "-log (10-6 cm/s)":
            pampa_val = np.power(10, -pampa_val + 7)
        case "log (10-6 cm/s)":
            pampa_val = np.power(10, pampa_val + 7)
        case "nm/s":
            pass
        case _:
            raise ValueError(f"Unknown unit: {unit}")
    return pampa_val


# Original code for descriptor functions adapted from
# https://github.com/brykimjh/degrader-permeability-ml3d-metaD/blob/main/data/calculate_2d_properties.py
def calculate_tnsa(mol):
    """Calculate Total Non-Polar Surface Area (TNSA)."""
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    total_surface_area = sum(
        20 if atom.GetAtomicNum() == 6 else 0 for atom in mol.GetAtoms()
    )
    tnsa = total_surface_area - tpsa
    return max(tnsa, 0)


# https://github.com/rdkit/rdkit/issues/1433
def calculate_charvol(mol):
    """Calculate Characteristic Volume (CharVol)."""
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    try:
        return AllChem.ComputeMolVolume(mol)
    except Exception as e:
        print(f"Error calculating CharVol for molecule {Chem.MolToSmiles(mol)}: {e}")
        return None


descriptor_functions = {
    "MolecularWeight": Descriptors.MolWt,
    "CharVol": calculate_charvol,
    "cLogD^7.4": Descriptors.MolLogP,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "RingCount": Descriptors.RingCount,
    "HydrogenBondAcceptorCount": Descriptors.NumHAcceptors,
    "HydrogenBondDonorCount": Descriptors.NumHDonors,
    "RotatableBondCount": Descriptors.NumRotatableBonds,
    "TopologicalPolarSurfaceArea": Descriptors.TPSA,
    "FractionCSP3": lambda mol: Descriptors.FractionCSP3(mol),
    "NumStereoCenters": lambda mol: len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    ),
    "AllBonds": lambda mol: mol.GetNumBonds(),
    "RingAtoms": lambda mol: sum(len(ring) for ring in mol.GetRingInfo().AtomRings()),
    "Halogens": lambda mol: sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]
    ),
    "HeteroAtoms": lambda mol: sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]
    ),
    "TNSA": calculate_tnsa,
    "Flexibility": lambda mol: (
        Descriptors.NumRotatableBonds(mol) / mol.GetNumBonds()
        if mol.GetNumBonds() > 0
        else 0
    ),
}


def calculate_properties(smiles: str) -> list[float | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_functions)
    result = [func(mol) for func in descriptor_functions.values()]
    return result


def butina_groups(smiles, threshold=0.35):
    gen = GetMorganGenerator(includeChirality=True)
    fingerprints = [gen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
    matrix = 1 - np.array(
        [DataStructs.BulkTanimotoSimilarity(fp, fingerprints) for fp in fingerprints]
    )

    clusters = Butina.ClusterData(
        data=matrix, nPts=len(fingerprints), distThresh=threshold, isDistData=True
    )
    clusters = sorted(clusters, key=len, reverse=True)
    groups = np.zeros(len(smiles), dtype=int)
    for cluster in clusters:
        groups[list(cluster)] = clusters.index(cluster)
    return groups, clusters
