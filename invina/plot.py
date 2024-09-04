from scipy.cluster.hierarchy import linkage
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdFMCS as rdFMCS

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


def plot_dendrogram(
    dist,
    linkage_method,
    leaf_data,
    width,
    height,
    title=None,
    count_sort=True,
    distance_sort=False,
    line_width=0.5,
    line_color='black',
    marker_size=15,
    leaf_color='CNN_score',
    render_mode='svg',
    leaf_y=0,
    leaf_color_discrete_map=None,
    leaf_category_orders=None,
    template='simple_white',
):

    import plotly.express as px
    import scipy.cluster.hierarchy as sch

    # Hierarchical clustering.
    Z = sch.linkage(dist, method=linkage_method)
    # Compute the dendrogram but don't plot it.
    dend = sch.dendrogram(
        Z,
        count_sort=count_sort,
        distance_sort=distance_sort,
        no_plot=True,
    )

    # Compile the line coordinates into a single dataframe.
    icoord = dend["icoord"]
    dcoord = dend["dcoord"]
    line_segments_x = []
    line_segments_y = []
    for ik, dk in zip(icoord, dcoord):
        # Adding None here breaks up the lines.
        line_segments_x += ik + [None]
        line_segments_y += dk + [None]
    df_line_segments = pd.DataFrame({"x": line_segments_x, "y": line_segments_y})

    # Convert X coordinates to haplotype indices (scipy multiplies coordinates by 10).
    df_line_segments["x"] = (df_line_segments["x"] - 5) / 10

    # Plot the lines.
    fig = px.line(
        df_line_segments,
        x="x",
        y="y",
        render_mode=render_mode,
        template=template,
    )

    # Reorder leaf data to align with dendrogram.
    leaves = dend["leaves"]
    n_leaves = len(leaves)
    leaf_data = leaf_data.iloc[leaves]

    # Add scatter plot to draw the leaves.
    fig.add_traces(
        list(
            px.scatter(
                data_frame=leaf_data,
                x=np.arange(n_leaves),
                y=np.repeat(leaf_y, n_leaves),
                color=leaf_color,
                render_mode=render_mode,
                hover_name='Mode',
                hover_data=leaf_data.columns.to_list(),
                template=template,
                color_discrete_map=leaf_color_discrete_map,
                category_orders=leaf_category_orders,
            ).select_traces()
        )
    )

    # Style the lines and markers.
    line_props = dict(
        width=line_width,
        color=line_color,
    )
    marker_props = dict(
        size=marker_size,
    )
    fig.update_traces(line=line_props, marker=marker_props)

    # Style the figure.
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        autosize=True,
        hovermode="closest",
        showlegend=False,
    )

    return fig, leaf_data

def plot_heatmap(df_rmsd, leaf_data):
    """Create an interactive heatmap using Plotly."""
    pose_order = leaf_data['Mode'].to_numpy() - 1
    
    fig = go.Figure(data=go.Heatmap(
        z=df_rmsd,
        x=np.arange(len(pose_order)),
        y=np.arange(len(pose_order)),
        colorscale='Viridis',
        showscale=False
    ))

    return fig

def _concat_subplots(
        figures,
        width,
        height,
    ):
        from plotly.subplots import make_subplots  # type: ignore
        # make subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.05,
            shared_xaxes=True
        )

        for i, figure in enumerate(figures):
            if isinstance(figure, go.Figure):
                # This is a figure, access the traces within it.
                for trace in range(len(figure["data"])):
                    fig.append_trace(figure["data"][trace], row=i+1, col=1)
            else:
                # Assume this is a trace, add directly.
                fig.append_trace(figure, row=i+1, col=1)

        fig.update_xaxes(visible=False)
        fig.update_layout(
            width=width,
            height=height,
            hovermode="closest",
            plot_bgcolor="white",
        )
        fig.update(layout_coloraxis_showscale=False)

        return fig