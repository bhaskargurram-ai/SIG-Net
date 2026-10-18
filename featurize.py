"""Chemprop-grade one-hot atom/bond featurization for PyG graphs.

The Phase 1 bundle stores PyG MoleculeNet integer atom codes (9 columns) and
integer bond codes (3 columns). Feeding those as continuous floats handicaps
every graph model. This module rebuilds x / edge_attr from the canonical
SMILES using the Chemprop v1 feature scheme (133 atom dims, 14 bond dims).
"""
import numpy as np
import torch
from rdkit import Chem

ATOM_LIST = list(range(1, 101))  # atomic numbers 1..100
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-1, -2, 1, 2, 0]
CHIRAL_TAGS = [0, 1, 2, 3]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _onehot_unk(value, choices):
    enc = [0.0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else -1
    enc[idx] = 1.0
    return enc


def atom_features(atom):
    f = (
        _onehot_unk(atom.GetAtomicNum(), ATOM_LIST)
        + _onehot_unk(atom.GetTotalDegree(), DEGREES)
        + _onehot_unk(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _onehot_unk(int(atom.GetChiralTag()), CHIRAL_TAGS)
        + _onehot_unk(int(atom.GetTotalNumHs()), NUM_HS)
        + _onehot_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetMass() * 0.01]
    )
    return f


ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))  # 133


def bond_features(bond):
    bt = bond.GetBondType()
    f = [
        0.0,  # not-None flag
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    f += _onehot_unk(int(bond.GetStereo()), list(range(6)))
    return f


BOND_FDIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))  # 14


def featurize_smiles(smiles: str):
    """Return (x [N, 133], edge_index [2, 2E], edge_attr [2E, 14]) as float/long tensors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        mol = Chem.MolFromSmiles("C")
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [bf, bf]
    if len(src) == 0:  # single heavy atom: self-loop with zero bond features
        src, dst, ea = [0], [0], [[0.0] * BOND_FDIM]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float32)
    return x, edge_index, edge_attr


def refeaturize_dataset(data_list):
    """In-place: replace x / edge_index / edge_attr on each PyG Data with Chemprop-grade features."""
    for d in data_list:
        x, ei, ea = featurize_smiles(d.smiles)
        d.x, d.edge_index, d.edge_attr = x, ei, ea
    return data_list


if __name__ == "__main__":
    x, ei, ea = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    print("ATOM_FDIM", ATOM_FDIM, "BOND_FDIM", BOND_FDIM, x.shape, ei.shape, ea.shape)
    assert x.shape == (13, ATOM_FDIM) and ei.shape[1] == 26 and ea.shape == (26, BOND_FDIM)
    print("ok")
