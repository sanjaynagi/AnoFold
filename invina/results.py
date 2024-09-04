import os
import re
import pickle
import pandas as pd
import numpy as np

import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .utils import pdb_to_residue_number
from .chem import calculate_distances
from .plot import _view_3d, plot_dendrogram, plot_heatmap, _concat_subplots

class Docked:
    def __init__(self, gene_id, ligand, wkdir="../", active_site_motif="[LIV].G.S.G", catalytic_codon_in_motif=4, catalytic_molecule="OG"):
        self.gene_id = gene_id
        self.ligand = ligand

        self.receptor_path = os.path.join(wkdir, f"receptors/{gene_id}.pdb")
        self.ligand_path = os.path.join(wkdir, f"ligands/{ligand}.pdbqt")
        self.docked_path = os.path.join(wkdir, f"docked/{gene_id}_{ligand}.sdf")
        self.log_path = os.path.join(wkdir, f"logs/{gene_id}_{ligand}.log")

        self.active_site_motif = active_site_motif
        self.catalytic_codon_in_motif = catalytic_codon_in_motif
        self.catalytic_molecule = catalytic_molecule

        self.df = self.load_docking_results() 
        self.values = self.df.to_numpy()
        self.poses = self.load_poses()


    def load_docking_results(self):
        
        residue_number = pdb_to_residue_number(receptor_path=self.receptor_path, active_site_motif=self.active_site_motif, catalytic_codon_in_motif=self.catalytic_codon_in_motif)
        # Calculate distances
        distances = calculate_distances(receptor_path=self.receptor_path, ligand_path=self.ligand_path, residue_number=residue_number, catalytic_molecule=self.catalytic_molecule)

        # Read gnina log file
        with open(self.log_path, 'r') as f:
            log_content = f.read()
        
        # Extract docking results
        pattern = r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
        docking_results = re.findall(pattern, log_content, re.MULTILINE)
        
        # Create DataFrame
        df = pd.DataFrame(docking_results, columns=['Mode', 'Affinity', 'Intramol', 'CNN_score', 'CNN_affinity'])
        df = df.astype({'Mode': int, 'Affinity': float, 'Intramol': float, 'CNN_score': float, 'CNN_affinity': float})
        
        # Add distances
        df['Distance'] = pd.Series(distances)
        
        # Reorder columns
        df = df[['Mode', 'Affinity', 'Intramol', 'CNN_score', 'CNN_affinity', 'Distance']]
        return df
    
    def load_poses(self):
        """Load poses from an SDF file and rename them."""
        poses = Chem.SDMolSupplier(self.docked_path)
        renamed_poses = []
        for index, p in enumerate(poses):
            if p is not None:
                p.SetProp('_Name', str(index + 1))
                renamed_poses.append(p)
        return renamed_poses

    def calculate_rmsd_matrix(self, poses):
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

    def plot_results(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create a subplot with 2 rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Docking Scores", "Distance to Active Site"))

        # Add traces for each score type
        for score in ['Affinity', 'CNN_score', 'CNN_affinity']:
            fig.add_trace(go.Scatter(x=self.df['Mode'], y=self.df[score], 
                                     mode='lines+markers', name=score), row=1, col=1)

        # Add trace for distance
        fig.add_trace(go.Scatter(x=self.df['Mode'], y=self.df['Distance'], 
                                 mode='lines+markers', name='Distance'), row=2, col=1)

        # Update layout
        fig.update_layout(title=f"Docking Results for {self.gene_id} with {self.ligand}",
                          xaxis_title="Mode", height=600)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Distance (Å)", row=2, col=1)

        return fig

    def view_3d(self, **kwargs):
        return _view_3d(**kwargs)
    
    def cluster_poses(self, width=1000, height=500):
        """Analyze poses and create visualizations."""
        from scipy.spatial.distance import squareform
        # Load poses
        poses = self.load_poses()
        
        # Calculate RMSD matrix
        rmsd_matrix = self.calculate_rmsd_matrix(poses)
        condensed_rmsd_matrix = squareform(rmsd_matrix, checks=False)
        # Create visualizations
        dendrogram, leaf_data = plot_dendrogram(dist=condensed_rmsd_matrix, linkage_method='complete', leaf_data=self.df, width=500, height=500)
        
        pose_order = leaf_data['Mode'].to_numpy() - 1
        df_rmsd = pd.DataFrame(rmsd_matrix).iloc[pose_order, pose_order]
        heatmap = plot_heatmap(df_rmsd, leaf_data)

        fig = _concat_subplots(figures=[dendrogram, heatmap], width=width, height=height)

        return fig, leaf_data

    def get_best_pose(self, metric='CNN_score'):
        return self.df.loc[self.df[metric].idxmax()]

    def to_csv(self, output_path, **kwargs):
        self.df.to_csv(output_path, index=False,  **kwargs)
    
    def to_excel(self, output_path):
        self.df.to_excel(output_path, index=False)

    def save(self, file_path):
        """
        Save the MolecularDockingResults object to a pickle file.
        
        Args:
        file_path (str): Path to save the pickle file
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Object successfully saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        """
        Load a MolecularDockingResults object from a pickle file.
        
        Args:
        file_path (str): Path to the pickle file
        
        Returns:
        MolecularDockingResults: The loaded object
        """
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Loaded object is not an instance of {cls.__name__}")
        return obj
    
    def __repr__(self):
            return f"MolecularDockingResults(gene_id='{self.gene_id}', ligand='{self.ligand}')"

    def _repr_html_(self):
        html = f"<h3>Molecular Docking Results for {self.gene_id} with {self.ligand}</h3>"
        html += self.df.to_html()
        # html += "<h4>Docking Scores Plot</h4>"
        # html += self.plot_results().to_html(full_html=False, include_plotlyjs='cdn')
        # # html += "<h4>3D Visualization</h4>"
        # # html += self.view_3d()._repr_html_()
        return html
