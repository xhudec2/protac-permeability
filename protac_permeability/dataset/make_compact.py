import os
import tempfile
from math import acos, degrees
from multiprocessing import Pool
from pathlib import Path

import mdtraj as md
from numpy import dot, linalg
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolfiles, rdmolops

XLIM1 = (6, 15)  # SASA plot
YLIM1 = None
XLIM2 = (4, 14)  # Rg plot
YLIM2 = None


def calculate_angle(v1, v2):
    cos_theta = dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return degrees(acos(cos_theta))


def find_backbone_atoms(mol, atom1_idx, atom2_idx):
    path = rdmolops.GetShortestPath(mol, atom1_idx, atom2_idx)
    return len(path) - 1 if path else 0


def find_intramolecular_hbonds(mol, confId=-1, eligibleAtoms=[7, 8], distTol=3.0):
    res = []
    conf = mol.GetConformer(confId)
    for i in range(mol.GetNumAtoms()):
        atomi = mol.GetAtomWithIdx(i)
        if atomi.GetAtomicNum() != 1 or atomi.GetDegree() != 1:
            continue
        donor_atom = atomi.GetNeighbors()[0]
        if donor_atom.GetAtomicNum() not in eligibleAtoms:
            continue
        donor_pos = conf.GetAtomPosition(donor_atom.GetIdx())
        hydrogen_pos = conf.GetAtomPosition(i)
        for j in range(mol.GetNumAtoms()):
            if j == i:
                continue
            atomj = mol.GetAtomWithIdx(j)
            if atomj.GetAtomicNum() not in eligibleAtoms or mol.GetBondBetweenAtoms(
                i, j
            ):
                continue
            acceptor_pos = conf.GetAtomPosition(j)
            dist = (hydrogen_pos - acceptor_pos).Length()
            if dist < distTol:
                v1 = hydrogen_pos - donor_pos
                v2 = hydrogen_pos - acceptor_pos
                angle = calculate_angle(v1, v2)
                ring_size = int(find_backbone_atoms(mol, donor_atom.GetIdx(), j) + 2)
                res.append((i + 1, j + 1, dist, angle, ring_size))
    return res


def calculate_radius_of_gyration_rdkit(mol):
    return rdMolDescriptors.CalcRadiusOfGyration(mol)


def calculate_sasa(mol, pdb_filename):
    Chem.rdmolfiles.MolToPDBFile(mol, pdb_filename)
    traj = md.load(pdb_filename)
    sasa = md.shrake_rupley(traj)
    total_sasa_nm2 = sasa.sum(axis=1)
    # convert nm² -> Å²
    return total_sasa_nm2[0] * 100.0


def calculate_3d_psa(mol):
    """Calculate 3D PSA (polar SASA) in Å² using mdtraj and polar atoms (N/O)."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        pdb_file = tmp.name

    rdmolfiles.MolToPDBFile(mol, pdb_file)

    traj = md.load(pdb_file)
    sasa = md.shrake_rupley(traj, mode="atom")  # nm² per atom
    os.remove(pdb_file)

    polar_atoms = [
        i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() in (7, 8)
    ]  # Nitrogen and Oxygen
    polar_sasa_nm2 = sasa[0, polar_atoms].sum()
    # nm² → Å²
    return polar_sasa_nm2 * 100.0


eligible_atoms = [7, 8]  # N and O
cutoff = 3.0  # H-bond cutoff distance


def make_compact(sdf_file):
    print("Using SDF file:", sdf_file)
    protac_name = sdf_file.stem
    print("Protac name:", protac_name)
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    max_ring_size = 0
    max_ring_conf = None

    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        hbonds = find_intramolecular_hbonds(
            mol, confId=-1, distTol=cutoff, eligibleAtoms=eligible_atoms
        )
        for _, _, _, _, ring_size in hbonds:
            if ring_size > max_ring_size:
                max_ring_size = ring_size
                max_ring_conf = idx

    print(
        f" Conformer index {max_ring_conf} has the largest ring size in H-bond: {max_ring_size} atoms"
    )

    pdb_out = f"compact_structures/{protac_name}.pdb"

    mol = supplier[max_ring_conf]
    mol = Chem.Mol(mol)
    Chem.MolToPDBFile(mol, pdb_out, confId=-1)
    print(f"Saved compact structure (max ring size) to: {pdb_out}")


if __name__ == "__main__":
    files = Path("all_protacs")
    with Pool(8) as pool:
        pool.map(make_compact, files.glob("*_qm.sdf"))
