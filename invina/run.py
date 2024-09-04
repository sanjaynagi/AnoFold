import os
import subprocess
from .prepare import download_and_prepare_ligand, download_and_prepare_alphafold_pdb, generate_motifs, pdb_to_active_site_coords
from .results import Docked

def dock(gene_id, ligand, override_motif=None, override_desc=None, wkdir="../", verbose=False):
    def log(message):
        if verbose:
            print(message)

    log(f"Starting docking process for gene_id: {gene_id}, ligand: {ligand}")

    pdb_path = os.path.join(wkdir, f"receptors/{gene_id}.pdbqt")
    ligand_path = os.path.join(wkdir, f"ligands/{ligand}.pdbqt")
    receptors_save_path = os.path.join(wkdir, "receptors/")
    ligand_save_path = os.path.join(wkdir, "ligands/")
    docked_folder_path = os.path.join(wkdir, "docked/")
    logs_folder_path = os.path.join(wkdir, "logs/")

    parent_dir = os.path.abspath(os.path.join(wkdir, os.pardir))
    if (os.path.exists(os.path.join(parent_dir, "docked")) and 
            os.path.exists(os.path.join(parent_dir, "logs"))):
        raise ValueError("There are docked and log directories in the parent directory of your specified working directory. "
                         "Are you sure you have specified the correct working directory?")
    
    os.makedirs(docked_folder_path, exist_ok=True)
    os.makedirs(logs_folder_path, exist_ok=True)

    log("Checking for receptor and ligand files...")
    if not os.path.exists(pdb_path):
        log(f"Receptor file not found. Downloading and preparing AlphaFold PDB for {gene_id}")
        download_and_prepare_alphafold_pdb(gene_id, output_dir=receptors_save_path, ph=7.4)
    
    if not os.path.exists(ligand_path):
        log(f"Ligand file not found. Downloading and preparing ligand {ligand}")
        download_and_prepare_ligand(ligand, save_path=ligand_save_path) 

    log("Generating or using provided motifs...")
    if override_motif:
        active_site_motif, catalytic_molecule, catalytic_codon_in_motif = override_motif
        log(f"Using provided override motif - active site motif: {active_site_motif}, catalytic molecule: {catalytic_molecule}, catalytic codon in motif: {catalytic_codon_in_motif}")
    else:
        active_site_motif, catalytic_molecule, catalytic_codon_in_motif = generate_motifs(gene_id=gene_id, 
                                                                                          override_desc=override_desc)
        log(f"Generated motifs- active site motif: {active_site_motif}, catalytic molecule: {catalytic_molecule}, catalytic codon in motif: {catalytic_codon_in_motif}")

    log("Getting active site coordinates...")
    res = pdb_to_active_site_coords(
        receptor_path=pdb_path,
        active_site_motif=active_site_motif, 
        catalytic_molecule=catalytic_molecule, 
        catalytic_codon_in_motif=catalytic_codon_in_motif)
    x, y, z = res[0]['coordinates']
    log(f"Active site coordinates: x={x}, y={y}, z={z}")

    log_path = f"{wkdir}logs/{gene_id}_{ligand}.log"
    if not os.path.exists(log_path) and os.path.exists(ligand_path):
        log("Preparing to run GNINA for docking...")
        command = [
            "./gnina",
            "-r", pdb_path,
            "-l", ligand_path,
            "--center_x", str(x),
            "--center_y", str(y),
            "--center_z", str(z),
            "--size_x", "20",
            "--size_y", "20",
            "--size_z", "20",
            "-o", os.path.join(docked_folder_path, f"{gene_id}_{ligand}.sdf"),
            "--log", log_path,
            "--seed", "0"
        ]
        
        log("Running GNINA command...")
        subprocess.run(command, check=True)
        log("GNINA docking completed successfully")
    else:
        log("Skipping GNINA docking: log file already exists or ligand file is missing")

    log("Docking process completed")
    return Docked(
                gene_id=gene_id, 
                ligand=ligand,
                wkdir=wkdir,
                active_site_motif=active_site_motif,
                catalytic_codon_in_motif=catalytic_codon_in_motif,
                catalytic_molecule=catalytic_molecule
                )