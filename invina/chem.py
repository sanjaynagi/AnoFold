import numpy as np

from rdkit import Chem
import rdkit.Chem.rdFMCS as rdFMCS

from .utils import pdb_to_pandas



def calculate_rmsd_matrix(poses):
    """Calculate RMSD matrix for all pose pairs."""
    import math
    size = len(poses)
    rmsd_matrix = np.zeros((size, size))
    
    for i, mol in enumerate(poses):
        for j, jmol in enumerate(poses):
            # MCS identification between reference pose and target pose
            r = rdFMCS.FindMCS([mol, jmol])
            # Atom map for reference and target
            a = mol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
            b = jmol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
            # Atom map generation
            amap = list(zip(a, b))

            # Calculate RMSD
            # distance calculation per atom pair
            distances=[]
            for atomA, atomB in amap:
                pos_A=mol.GetConformer().GetAtomPosition (atomA)
                pos_B=jmol.GetConformer().GetAtomPosition (atomB)
                coord_A=np.array((pos_A.x,pos_A.y,pos_A.z))
                coord_B=np.array ((pos_B.x,pos_B.y,pos_B.z))
                dist_numpy = np.linalg.norm(coord_A-coord_B)
                distances.append(dist_numpy)
    
            # This is the RMSD formula from wikipedia
            rmsd=math.sqrt(1/len(distances)*sum([i*i for i in distances]))
    
            #saving the rmsd values to a matrix and a table for clustering
            rmsd_matrix[i ,j]=rmsd
    
    return rmsd_matrix


def calculate_distances(receptor_path, ligand_path, residue_number, catalytic_molecule):
    # Load the protein structure using our pandas loader
    receptor_df = pdb_to_pandas(receptor_path)
    # Find the specific atom in the protein
    target_atom = receptor_df[(receptor_df['residue_number'] == residue_number) & 
                              (receptor_df['atom_name'] == catalytic_molecule)]
    
    if target_atom.empty:
        raise ValueError(f"Atom {catalytic_molecule} not found in residue {residue_number}")
    
    target_coords = np.array([target_atom['x'].values[0], 
                              target_atom['y'].values[0], 
                              target_atom['z'].values[0]])
    
    # Load the ligand with all conformations
    suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
    
    distances = []
    for mol in suppl:
        if mol is None:
            distances.append(None)
            continue
        
        # Find the ester bond in the ligand
        ester_bond = find_ester_bond(mol)
        
        if ester_bond is None:
            distances.append(None)
            continue
        
        # Calculate the midpoint of the ester bond
        conf = mol.GetConformer()
        pos1 = conf.GetAtomPosition(ester_bond.GetBeginAtomIdx())
        pos2 = conf.GetAtomPosition(ester_bond.GetEndAtomIdx())
        midpoint = np.array([(pos1.x + pos2.x) / 2, (pos1.y + pos2.y) / 2, (pos1.z + pos2.z) / 2])
        
        # Calculate distance
        distance = np.linalg.norm(target_coords - midpoint)
        distances.append(distance)
    
    return distances



def find_ester_bond(mol):
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        # Check if the bond is between carbon and oxygen
        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 8) or \
           (atom2.GetAtomicNum() == 6 and atom1.GetAtomicNum() == 8):
            # Identify which atom is carbon and which is oxygen
            c_atom = atom1 if atom1.GetAtomicNum() == 6 else atom2
            o_atom = atom2 if atom1.GetAtomicNum() == 6 else atom1
            
            # Check if the carbon is connected to another oxygen (double-bonded)
            for neighbor in c_atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 8 and neighbor.GetIdx() != o_atom.GetIdx():
                    return bond
    return None