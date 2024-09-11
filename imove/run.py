import os
import subprocess
from .utils import prepare_folders, log
from .prepare import (
    download_and_prepare_ligand, 
    download_and_prepare_alphafold_pdb, 
    generate_motifs, 
    pdb_to_active_site_coords,
    pdb_to_heme_coords
)
from .results import Docked

def dock(gene_id, ligand, override_motif=None, override_desc=None, mutagenesis_dict=None, wkdir="../", verbose=False, box_size=20, p450=False):

    log(f"Starting docking process for gene_id: {gene_id}, ligand: {ligand}", verbose=verbose)

    if mutagenesis_dict:
        mut_str = "_" + '_'.join([str(key) + value for key, value in mutagenesis_dict.items()])
    else:
        mut_str = ""

    pdb_path, ligand_path, receptors_save_path, ligand_save_path, docked_folder_path, logs_folder_path = prepare_folders(wkdir, gene_id, ligand, mut_str)

    if not os.path.exists(pdb_path):
        log(f"Receptor file not found. Downloading and preparing AlphaFold PDB for {gene_id}", verbose=verbose)
        download_and_prepare_alphafold_pdb(gene_id, output_dir=receptors_save_path, ph=7.4, mutagenesis_dict=mutagenesis_dict, p450=p450)
    else:
        log("receptor file found...", verbose=verbose)

    if not os.path.exists(ligand_path):
        log(f"Ligand file not found. Downloading and preparing ligand {ligand}", verbose=verbose)
        download_and_prepare_ligand(ligand, save_path=ligand_save_path) 

    if p450:
        active_site_motif, catalytic_molecule, catalytic_codon_in_motif = None, None, None
        x, y, z = pdb_to_heme_coords(pdb_path)        
    else:
        if override_motif:
            active_site_motif, catalytic_molecule, catalytic_codon_in_motif = override_motif
            log(f"Using provided override motif - active site motif: {active_site_motif}, catalytic molecule: {catalytic_molecule}, catalytic codon in motif: {catalytic_codon_in_motif}", verbose=verbose)
        else:
            active_site_motif, catalytic_molecule, catalytic_codon_in_motif = generate_motifs(gene_id=gene_id, wkdir=wkdir,
                                                                                            override_desc=override_desc)
            log(f"Generated motifs - active site motif: {active_site_motif}, catalytic molecule: {catalytic_molecule}, catalytic codon in motif: {catalytic_codon_in_motif}", verbose=verbose)

        res = pdb_to_active_site_coords(
            receptor_path=pdb_path,
            active_site_motif=active_site_motif, 
            catalytic_molecule=catalytic_molecule, 
            catalytic_codon_in_motif=catalytic_codon_in_motif)
        x, y, z = res[0]['coordinates']

    log(f"Active site coordinates: x={x}, y={y}, z={z}", verbose=verbose)

    log_path = os.path.join(logs_folder_path, f"{gene_id}{mut_str}_{ligand}.log")
    if not os.path.exists(log_path) and os.path.exists(ligand_path):
        log("Preparing to run GNINA for docking...", verbose=verbose)
        command = [
            "./gnina",
            "-r", pdb_path,
            "-l", ligand_path,
            "--center_x", str(x),
            "--center_y", str(y),
            "--center_z", str(z),
            "--size_x", f"{box_size}",
            "--size_y", f"{box_size}",
            "--size_z", f"{box_size}",
            "-o", os.path.join(docked_folder_path, f"{gene_id}{mut_str}_{ligand}.sdf"),
            "--log", log_path,
            "--seed", "0"
        ]
        
        subprocess.run(command, check=True)
    else:
        log("Skipping GNINA docking: log file already exists or ligand file is missing", verbose=verbose)

    log("Docking process completed")
    return Docked(
                gene_id=gene_id, 
                ligand=ligand,
                wkdir=wkdir,
                active_site_motif=active_site_motif,
                catalytic_codon_in_motif=catalytic_codon_in_motif,
                catalytic_molecule=catalytic_molecule, 
                mutagenesis_dict=mutagenesis_dict,
                p450=p450
                )



def download_pdb(gene_id, verbose=False, mutagenesis_dict=None, wkdir="../", p450=False):

    log(f"Downloading and preparing PDB file for gene_id: {gene_id}", verbose=verbose)

    if mutagenesis_dict:
        mut_str = "_" + '_'.join([str(key) + value for key, value in mutagenesis_dict.items()])
    else:
        mut_str = ""

    pdb_path = os.path.join(wkdir, f"receptors/{gene_id}{mut_str}.pdbqt")
    receptors_save_path = os.path.join(wkdir, "receptors/")

    if not os.path.exists(pdb_path):
        log(f"Receptor file not found. Downloading and preparing AlphaFold PDB for {gene_id}", verbose=verbose)
        download_and_prepare_alphafold_pdb(gene_id, output_dir=receptors_save_path, ph=7.4, mutagenesis_dict=mutagenesis_dict, p450=p450)
    else:
        log("Receptor file found...", verbose=verbose)
    
