import numpy as np
import pandas as pd
import re
import requests
import time


amino_acid_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

def log(message, verbose=False):
    if verbose:
        print(message)

def prepare_folders(wkdir, gene_id, ligand, mut_str):
    import os

    pdb_path = os.path.join(wkdir, f"receptors/{gene_id}{mut_str}.pdbqt")
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
    return pdb_path, ligand_path, receptors_save_path, ligand_save_path, docked_folder_path, logs_folder_path


def parse_pdb(file_path):
    columns = ['record_type', 'atom_number', 'atom_name', 'alt_loc', 'residue_name', 'chain_id', 
               'residue_number', 'insertion', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'element', 'charge']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                record_type = line[0:6].strip()
                atom_number = int(line[6:11])
                atom_name = line[12:16].strip()
                alt_loc = line[16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21].strip()
                residue_number = int(line[22:26])
                insertion = line[26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60])
                temp_factor = float(line[60:66])
                element = line[76:78].strip()
                charge = line[78:80].strip()

                data.append([record_type, atom_number, atom_name, alt_loc, residue_name, chain_id, 
                             residue_number, insertion, x, y, z, occupancy, temp_factor, element, charge])

    return pd.DataFrame(data, columns=columns).sort_values('residue_number')

def parse_pdbqt(file_path):
    columns = ['record_type', 'atom_number', 'atom_name', 'alt_loc', 'residue_name', 'chain_id', 
               'residue_number', 'insertion', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'partial_charge', 'atom_type']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                record_type = line[0:6].strip()
                atom_number = int(line[6:11])
                atom_name = line[12:16].strip()
                alt_loc = line[16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21].strip()
                residue_number = int(line[22:26])
                insertion = line[26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60])
                temp_factor = float(line[60:66])
                partial_charge = float(line[70:76])
                atom_type = line[77:79].strip()

                data.append([record_type, atom_number, atom_name, alt_loc, residue_name, chain_id, 
                             residue_number, insertion, x, y, z, occupancy, temp_factor, partial_charge, atom_type])

    return pd.DataFrame(data, columns=columns).sort_values('residue_number')

def pdb_to_pandas(file_path):
    if file_path.lower().endswith('.pdb'):
        return parse_pdb(file_path)
    elif file_path.lower().endswith('.pdbqt'):
        return parse_pdbqt(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pdb or .pdbqt file.")

def pdb_to_residue_number(receptor_path, active_site_motif, catalytic_codon_in_motif):
    df_pdb = pdb_to_pandas(receptor_path)
    
    # Extract unique amino acids
    aa3 = df_pdb[['residue_name', 'residue_number']].drop_duplicates()['residue_name']
    convert = np.vectorize(lambda x: amino_acid_map.get(x, x))
    aa1 = convert(aa3)
    
    # Find all occurrences of the motif
    aa_sequence = ''.join(aa1)
    motif_matches = list(re.finditer(active_site_motif, aa_sequence))

    for match in motif_matches:
        start, end = match.start(), match.end()
        # Find coordinates for the target molecule
        target_idx = start + 1 + catalytic_codon_in_motif

    return target_idx



def vectorbase_to_uniprot(gene_id):
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": "VEuPathDB",
        "to": "UniProtKB",
        "ids": f"VectorBase:{gene_id}"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()
    job_id = response.json()["jobId"]

    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        status_response = requests.get(status_url)
        status_response.raise_for_status()
        status = status_response.json()
        if "jobStatus" in status and status["jobStatus"] in ("RUNNING", "NEW"):
            continue
        elif "results" in status or "failedIds" in status:
            break

    results_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
    results_response = requests.get(results_url)
    results_response.raise_for_status()
    results = results_response.json()['results']

    # if multiple transcripts, choose the longest
    lens = {}
    if len(results) > 1:
        for entry in results:
            gene_data = get_uniprot_data(entry['to'], vectorbase=False)
            lens[entry['to']] = gene_data['length']

        max_key = max(lens, key=lens.get)
        print(f"selecting longest transcript for gene_id: {gene_id}, {max_key} : {lens[max_key]}")
        return max_key
    else:
        return results[0]["to"]


def get_uniprot_data(gene_id, vectorbase=True):
    if vectorbase:
        uniprot_acc = vectorbase_to_uniprot(gene_id)
    else:
        uniprot_acc = gene_id
    if not uniprot_acc:
        return f"No UniProt accession found for VectorBase ID: {gene_id}"

    # UniProt API endpoint
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    print(f"UniProt accession: {uniprot_acc}")

    # Query parameters
    params = {
        "query": f"accession:{uniprot_acc}",
        "format": "json",
        "fields": "gene_names,protein_name,organism_name,go, sequence"
    }

    # Send request with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['results']:
                result = data['results'][0]
                return {
                    "gene_name": result.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A'),
                    "protein_name": result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A'),
                    "organism": result.get('organism', {}).get('scientificName', 'N/A'),
                    "function": next((comment['texts'][0]['value'] for comment in result.get('comments', []) if comment['commentType'] == 'FUNCTION'), 'N/A'),
                    "go_terms": [go['id'] for go in result.get('goTerms', [])],
                    "length": result.get('sequence', {}).get('length', 'N/A')  # Added this line to get the length
                }
            else:
                return f"No data found for UniProt accession: {uniprot_acc}"

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"Error retrieving data: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

    return "Max retries reached. Unable to retrieve data."
