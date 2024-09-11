import numpy as np
from rdkit import Chem
from typing import List, Tuple

from .utils import pdb_to_pandas



def get_atoms_midpoint(conf, atom_indices: List[int]) -> Tuple[float, float, float]:
    """
    Calculate the 3D midpoint of any number of atoms.
    
    :param conf: RDKit conformer object
    :param atom_indices: List of atom indices
    :return: Tuple representing the 3D coordinates of the midpoint
    """
    if not atom_indices:
        raise ValueError("No atom indices provided")
    
    total_x, total_y, total_z = 0, 0, 0
    for idx in atom_indices:
        pos = conf.GetAtomPosition(idx)
        total_x += pos.x
        total_y += pos.y
        total_z += pos.z
    
    num_atoms = len(atom_indices)
    midpoint = (total_x / num_atoms, total_y / num_atoms, total_z / num_atoms)
    return midpoint



def measure_distance_to_atoms(mol, residue_coords, smarts_pattern, conf_id=-1):
    """
    Measure the distance from a residue to atoms specified by SMARTS.
    
    :param mol: RDKit molecule object
    :param residue_coords: Tuple of (x, y, z) coordinates of the residue
    :param smarts_pattern: SMARTS pattern specifying the atoms of interest
    :param conf_id: Conformer ID to use (-1 for default conformer)
    :return: Distance from the residue to the midpoint of matching atoms
    """
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern))
    if not matches:
        print(f"Warning: No matches found for pattern {smarts_pattern}")
        return None
    
    conf = mol.GetConformer(conf_id)
    distances = []
    
    for match in matches:
        midpoint = get_atoms_midpoint(conf, match)
        distance = sum((a - b) ** 2 for a, b in zip(residue_coords, midpoint)) ** 0.5
        distances.append(distance)
    
    return min(distances) if distances else None


def calculate_distances(receptor_path, docked_sdf_path, residue_number, catalytic_molecule, smarts_pattern, p450=False):
    """
    Process all conformations in an SDF file and measure distances to specified atoms.
    
    :param sdf_file: Path to the SDF file
    :param residue_coords: Tuple of (x, y, z) coordinates of the residue
    :param smarts_pattern: SMARTS pattern specifying the atoms of interest
    :return: List of distances for each conformation
    """
    suppl = Chem.SDMolSupplier(docked_sdf_path, removeHs=False)

    # Load the protein structure using our pandas loader
    receptor_df = pdb_to_pandas(receptor_path)

    if p450:
        target_atom = receptor_df.query("atom_name == 'FE'")
    else:
        # Find the specific atom in the protein
        target_atom = receptor_df[
            (receptor_df["residue_number"] == residue_number)
            & (receptor_df["atom_name"] == catalytic_molecule)
        ]

    if target_atom.empty:
        raise ValueError(
            f"Atom {catalytic_molecule} not found in residue {residue_number}"
        )

    target_coords = target_atom[["x", "y", "z"]].values[0]

    all_distances = []

    for mol in suppl:
        if mol is not None:
            for conf_id in range(mol.GetNumConformers()):
                distance = measure_distance_to_atoms(mol, target_coords, smarts_pattern, conf_id)
                all_distances.append(distance)
    
    return all_distances, smarts_pattern



# Dictionary of SMARTS patterns for common bonds and structural features
def smarts_patterns():
    return {
    # Single bonds
    "alkyl_bond": "C-C",
    "ether_bond": "[#6]-[#8]-[#6]",
    "amine_bond": "[#6]-[#7]",
    "hydroxyl_bond": "[#6]-[#8]-[#1]",
    
    # Double bonds
    "alkene_bond": "C=C",
    "carbonyl_bond": "C=O",
    "imine_bond": "C=N",
    
    # Triple bonds
    "alkyne_bond": "C#C",
    "nitrile_bond": "C#N",
    
    # Aromatic bonds
    "aromatic_bond": "cc",
    
    # Functional groups
    "ester_group": "[#6]-C(=O)-O-[#6]",
    "amide_group": "[#6]-C(=O)-N",
    "carboxylic_acid": "[#6]-C(=O)-O-[#1]",
    "sulfonamide_group": "S(=O)(=O)-N",
    "phosphate_group": "P(=O)(-O)(-O)-O",
    
    # Ring systems
    "benzene_ring": "c1ccccc1",
    "pyridine_ring": "n1ccccc1",
    "pyrimidine_ring": "n1cnccc1",
    "imidazole_ring": "n1cncc1",
    "piperidine_ring": "N1CCCCC1",
    "morpholine_ring": "O1CNCCC1",
    
    # Specific positions in rings
    "benzene_para": "c1ccc(cc1)*",
    "benzene_meta": "c1cccc(c1)*",
    "benzene_ortho": "c1ccccc1*",
    
    # Heterocycles
    "furan_ring": "o1cccc1",
    "thiophene_ring": "s1cccc1",
    "pyrrole_ring": "[nH]1cccc1",
    
    # Specific groups
    "methyl_group": "C[#6]",
    "hydroxyl_group": "[OX2H]",
    "amino_group": "[NX3;H2,H1;!$(NC=O)]",
    "nitro_group": "[N+](=O)[O-]",
    
    # Halides
    "chloro_group": "Cl",
    "fluoro_group": "F",
    "bromo_group": "Br",
    "iodo_group": "I",
    
    # Specific bond types in context
    "amide_c_n_bond": "C(=O)-N",
    "ester_c_o_bond": "C(=O)-O",
    "ether_c_o_c_bond": "C-O-C",
    
    # More complex patterns
    "benzyl_group": "c1ccccc1-C",
    "phenol_group": "c1ccccc1-[OX2H]",
    "carbamate_group": "[NX3][CX3](=[OX1])[OX2]",
    "guanidino_group": "[NX3][CX3](=[NX2])[NX3]",
}
