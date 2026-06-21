import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    lfg = rdMolStandardize.LargestFragmentChooser()
    mol = lfg.choose(mol)
    return Chem.MolToSmiles(mol)


def try_float(pampa):
    try:
        if isinstance(pampa, str) and pampa[0] == "<":
            pampa = pampa[1:]
        pampa_val = float(pampa)
        return pampa_val
    except Exception as _:
        return np.nan


def calculate_tnsa(mol):
    """Calculate Total Non-Polar Surface Area (TNSA)."""
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    total_surface_area = sum(
        20 if atom.GetAtomicNum() == 6 else 0 for atom in mol.GetAtoms()
    )
    tnsa = total_surface_area - tpsa
    return max(tnsa, 0)


descriptor_functions = {
    "MolecularWeight": Descriptors.MolWt,
    "ExactMass": Descriptors.ExactMolWt,
    "XLogP3": Descriptors.MolLogP,
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
    "SizeShape": lambda mol: sum(len(ring) for ring in mol.GetRingInfo().AtomRings()),
    "Flexibility": lambda mol: Descriptors.NumRotatableBonds(mol) / mol.GetNumBonds()
    if mol.GetNumBonds() > 0
    else 0,
}


def calculate_properties(smiles: str) -> list[float | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_functions)
    result = [func(mol) for func in descriptor_functions.values()]
    return result
