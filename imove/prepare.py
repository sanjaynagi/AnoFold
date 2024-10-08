from openbabel import openbabel
import numpy as np
import pandas as pd
import os
import re
import requests

from .utils import (
    vectorbase_to_uniprot,
    get_uniprot_data,
    pdb_to_pandas,
    amino_acid_map,
)


def download_ligand(
    ligand_name,
    repo_url="https://raw.githubusercontent.com/sanjaynagi/imove/main/ligands/raw",
    save_path="ligands",
):
    os.makedirs(save_path, exist_ok=True)

    file_url = os.path.join(repo_url, f"{ligand_name}.pdbqt")
    file_path = os.path.join(save_path, f"{ligand_name}.pdbqt")

    try:
        response = requests.get(file_url)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Ligand '{ligand_name}.pdbqt' downloaded successfully.")
        return file_path

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Ligand file '{ligand_name}.pdbqt' not found at {file_url}"
            )
        else:
            raise ValueError(f"HTTP error occurred while downloading ligand: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error occurred while downloading ligand: {e}")


def download_and_prepare_ligand(
    ligand_name,
    save_path,
    repo_url="https://raw.githubusercontent.com/sanjaynagi/imove/main/ligands/raw",
    pH=7.4,
):
    """
    Download a ligand file and prepare it by adding hydrogens at a specified pH using OpenBabel.

    Parameters:
    - ligand_name (str): The name of the ligand (without the .pdbqt extension).
    - repo_url (str): The base URL of the GitHub repository's raw content.
    - save_path (str): The local directory where the files will be saved.
    - pH (float): The pH at which to add hydrogens. Default is 7.4.

    Returns:
    - str: Path to the prepared ligand file.
    """
    download_ligand(ligand_name, repo_url, save_path)

    # Set up OpenBabel conversion
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdbqt")

    mol = openbabel.OBMol()

    # Read the downloaded PDBQT file
    input_path = os.path.join(save_path, "raw", f"{ligand_name}.pdbqt")
    obConversion.ReadFile(mol, input_path)

    # Add hydrogens at the specified pH
    mol.AddHydrogens(False, True, pH)

    # Write the prepared molecule to a new file
    output_path = os.path.join(save_path, f"{ligand_name}.pdbqt")
    obConversion.WriteFile(mol, output_path)

    print(f"Ligand prepared and saved as '{output_path}'")
    return output_path



def download_and_prepare_alphafold_pdb(gene_id, output_dir, ph=7.4, mutagenesis_dict=None, p450=False):
    # Convert VectorBase GeneID to UniProt accession
    uniprot_accession = vectorbase_to_uniprot(gene_id)
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_accession}-F1-model_v4.pdb"
    )
    if response.status_code != 200:
        print(
            f"Failed to download AlphaFold PDB for {gene_id} (UniProt: {uniprot_accession}). Status code: {response.status_code}"
        )
        return None

    raw_pdb_file = os.path.join(output_dir, f"{gene_id}_raw.pdb")
    with open(raw_pdb_file, "wb") as f:
        f.write(response.content)

    # if we want to mutate, do it now before adding hydrogens and pdbqt 
    mut_str = ""
    if mutagenesis_dict:
        raw_pdb_file, mut_str = mutate_residue(receptor_path=raw_pdb_file, mutagenesis_dict=mutagenesis_dict)

    if p450:
        # Add heme to P450 structure
        raw_pdb_file = add_heme_to_p450(receptor_path=raw_pdb_file, receptor_with_heme_out_path=raw_pdb_file)
                                        
    output_file = os.path.join(output_dir, f"{gene_id}{mut_str}.pdbqt")
    # Use OpenBabel to add hydrogens
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, raw_pdb_file)
    # Add hydrogens
    mol.AddHydrogens(False, True, ph)
    # Write the processed molecule to a new PDB file
    obConversion.WriteFile(mol, output_file)

    print(
        f"Downloaded and protonated AlphaFold PDB for {gene_id} (UniProt: {uniprot_accession}) to {output_file}"
    )

def pdb_to_heme_coords(pdb_path):
    df_pdb = pdb_to_pandas(pdb_path)
    x,y,z = df_pdb.query("atom_name == 'FE'")[['x', 'y', 'z']].values[0]
    return x, y, z


def pdb_to_active_site_coords(
    receptor_path,
    active_site_motif,
    catalytic_codon_in_motif,
    catalytic_molecule,
    flanking_size=5,
):
    df_pdb = pdb_to_pandas(receptor_path)
    aa3 = df_pdb[["residue_name", "residue_number"]].drop_duplicates()["residue_name"]

    convert = np.vectorize(lambda x: amino_acid_map.get(x, x))
    aa1 = convert(aa3)

    # Find all occurrences of the motif
    aa_sequence = "".join(aa1)
    motif_matches = list(re.finditer(active_site_motif, aa_sequence))

    results = []
    for match in motif_matches:
        start, end = match.start(), match.end()

        # Get flanking region
        flanking_start = max(0, start - flanking_size)
        flanking_end = min(len(aa_sequence), end + flanking_size)
        flanking = (
            aa_sequence[flanking_start:start]
            + "["
            + aa_sequence[start:end]
            + "]"
            + aa_sequence[end:flanking_end]
        )

        print(
            f"Motif detected in PDB at codon {start+1}:{end} = {aa_sequence[start:end]}, Flanking sequence: {flanking}"
        )

        # Find coordinates for the target molecule
        target_idx = start + 1 + catalytic_codon_in_motif
        coords = df_pdb.query(
            f"residue_number == {target_idx} and atom_name == '{catalytic_molecule}'"
        )

        if not coords.empty:
            coord_array = (
                coords.drop_duplicates("atom_name")[["x", "y", "z"]]
                .to_numpy()[0]
                .astype(float)
            )
            results.append(
                {
                    "start": start + 1,
                    "end": end,
                    "motif": aa_sequence[start:end],
                    "flanking": flanking,
                    "coordinates": coord_array,
                    "molecule_number": int(coords["atom_number"].values[0]),
                }
            )

    if len(results) > 1:
        print(f"Warning, multiple {len(results)} matching motifs found")

    return results


def generate_motifs(gene_id, wkdir, override_desc):
    gene_data = get_uniprot_data(gene_id)
    gene_desc = gene_data["protein_name"]
    gene_name = gene_data["gene_name"]

    if override_desc:
        gene_desc = override_desc

    print(f"Uniprot info: {gene_id} | {gene_name} ({gene_desc})")
    if "ester hydrolase" in gene_desc or "Carboxylesterase" in gene_desc:
        return ("[LIV].G.S.G", "OG", 4)
    elif "glutathione transferase" in gene_desc:
        df_motif = pd.read_csv(f"{wkdir}/resources/AgamP4_gst_motifs.csv").set_index("GeneID")
        motif = df_motif.loc[gene_id, "motif"]
        return (motif, "O", 4)
    else:
        assert f"Unknown gene family: {gene_desc}, custom motif required"


def mutate_residue(receptor_path, mutagenesis_dict, pack_radius = 1):
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.toolbox import mutants

    pyrosetta.init(
        "-write_pdb_title_section_records true "
        "-write_pdb_link_records true "
        "-use_pdb_format_HETNAM_records true "
        "-write_pdb_parametric_info true "
        "-write_glycan_pdb_codes false "
        "-write_seqres_records true "
        "-output_pose_energies_table false"
    )
    pose = pose_from_pdb(receptor_path)

    for mutant_position, mutant_aa in mutagenesis_dict.items():
        mutants.mutate_residue(pose, mutant_position=mutant_position, mutant_aa=mutant_aa, pack_radius=pack_radius)
        print(f"Mutated residue {mutant_position} to {mutant_aa}")
    
    mut_str = "_" + '_'.join([str(key) + value for key, value in mutagenesis_dict.items()])
    pdb_mutant_path = receptor_path.replace(".pdb", f"{mut_str}.pdb")
    pose.dump_pdb(pdb_mutant_path)
    print(f"Mutagenesis complete. New structure saved as '{pdb_mutant_path}'")

    return pdb_mutant_path, mut_str




def download_ref_pdb(pdb_id='1TQN', save_dir='.'):
    """
    Download a PDB file from the RCSB PDB database.
    
    Args:
    pdb_id (str): The 4-character PDB ID of the structure to download.
    save_dir (str): The directory to save the downloaded file. Defaults to the current directory.
    
    Returns:
    str: The path to the downloaded file if successful, None otherwise.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Successfully downloaded {pdb_id}.pdb to {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {pdb_id}.pdb: {e}")
        return None
    
def copy_heme_to_p450(reference_pdb_path, my_pdb_path, receptor_with_heme_out_path=None):
    """
    Copy all HEM-related coordinates from a reference PDB to a P450 PDB file.
    
    Args:
    reference_pdb_path (str): Path to the reference PDB file containing the heme.
    p450_pdb_path (str): Path to the P450 PDB file to modify.
    output_pdb_path (str, optional): Path to save the modified P450 PDB file. If None, overwrites the input file.
    
    Returns:
    str: The path to the output PDB file.
    """
    # Read all HEM lines from the reference PDB
    hem_lines = []
    with open(reference_pdb_path, 'r') as ref_file:
        for line in ref_file:
            if line.startswith("HETATM") and "HEM" in line:
                hem_lines.append(line)
    
    if not hem_lines:
        raise ValueError("No HEM atoms found in the reference PDB file")

    # Read the P450 PDB file
    with open(my_pdb_path, 'r') as p450_file:
        p450_lines = p450_file.readlines()

    # Find the insertion point (before the last TER or END line)
    insert_index = len(p450_lines)
    for i in range(len(p450_lines) - 1, -1, -1):
        if p450_lines[i].strip().startswith(("TER", "END")):
            insert_index = i
            break

    # Insert all HEM lines
    p450_lines[insert_index:insert_index] = hem_lines

    # Write the modified content to a new file or overwrite the existing one
    output_path = receptor_with_heme_out_path or my_pdb_path
    with open(output_path, 'w') as output_file:
        output_file.writelines(p450_lines)

    print(f"HEM coordinates inserted successfully. Output written to {output_path}")
    print(f"Number of HEM lines inserted: {len(hem_lines)}")
    return output_path


def add_heme_to_p450(receptor_path, receptor_with_heme_out_path, ref_pdb_id="1TQN"):
    """
    Add heme to a P450 structure using ChimeraX and manual coordinate insertion.
    
    Args:
    receptor_path (str): Path to the input P450 PDB file.
    receptor_with_heme_path (str): Path to save the P450 with added heme.
    ref_pdb_id (str): PDB ID of the reference structure (default: "1TQN").
    
    Returns:
    str: Path to the final P450 PDB file with heme added.
    """
    import subprocess
    ref_p450_path = download_ref_pdb(ref_pdb_id)
    if not ref_p450_path:
        raise ValueError(f"Failed to download reference PDB {ref_pdb_id}")

    chimerax_script = f"""
    open {receptor_path}
    open {ref_p450_path}
    match #1 to #2
    close #2
    save {receptor_with_heme_out_path} models #1
    exit
    """

    script_path = "temp_chimerax_script.cxc"
    with open(script_path, "w") as script_file:
        script_file.write(chimerax_script)

    try:
        subprocess.run(["chimerax", "--nogui", script_path], check=True)
        print(f"ChimeraX alignment completed. Result saved to {receptor_with_heme_out_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running ChimeraX: {str(e)}")
        raise
    finally:
        os.remove(script_path)

    final_path = copy_heme_to_p450(reference_pdb_path=ref_p450_path, my_pdb_path=receptor_path, receptor_with_heme_out_path=receptor_with_heme_out_path)
    return final_path