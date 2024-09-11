import plotly.graph_objects as go
import numpy as np
import pandas as pd
import py3Dmol


def _view_3d(
    receptor_path,
    ligand_path=None,
    docked_sdf_path=None,
    receptor_highlight=None,
    style="cartoon",
    surface=False,
    surface_opacity=0.8,
    surface_color="white",
    highlight_color="yellow",
    stick_radius=0.1,
    zoom_ligand=True,
    rotate_degree=90,
    animation_interval=5000,
):
    """
    Enhanced 3D viewer for protein structures with optional ligand and docking results.

    Parameters:
    - receptor_path (str): Path to the receptor PDB file
    - ligand_path (str, optional): Path to the ligand PDB file
    - docked_sdf_path (str, optional): Path to the docked pose PDB file
    - receptor_highlight (int, optional): Residue number to highlight in the receptor
    - style (str): Style for receptor visualization ('cartoon', 'stick', 'sphere', or 'line')
    - surface (bool): Whether to show the protein surface
    - surface_opacity (float): Opacity of the surface (0.0 to 1.0)
    - surface_color (str): Color of the surface
    - highlight_color (str): Color for highlighted residues
    - stick_radius (float): Radius for stick representation
    - zoom_ligand (bool): Whether to zoom to the ligand
    - rotate_degree (int): Degrees to rotate the view
    - animation_interval (int): Interval for animation in milliseconds

    Returns:
    - py3Dmol.view: The 3D view object
    """
    v = py3Dmol.view()

    # Add receptor
    v.addModel(open(receptor_path).read())

    # Set receptor style
    style_dict = {style: {}}
    if style == "stick":
        style_dict["stick"]["radius"] = stick_radius
    v.setStyle({"model": -1}, style_dict)

    # Add surface if requested
    if surface:
        v.addSurface(py3Dmol.VDW, {"opacity": surface_opacity, "color": surface_color})

    # Highlight receptor residues if specified
    if receptor_highlight:
        for i in range(receptor_highlight - 3, receptor_highlight + 4):
            v.setStyle(
                {"model": -1, "resi": i},
                {
                    style: {"color": highlight_color},
                    "stick": {"radius": stick_radius * 2, "color": highlight_color},
                },
            )

    # Add ligand if provided
    if ligand_path:
        v.addModel(open(ligand_path).read())
        v.setStyle(
            {"model": -2},
            {"stick": {"colorscheme": "dimgrayCarbon", "radius": stick_radius * 1.25}},
        )

    # Add docked poses if provided
    if docked_sdf_path:
        v.addModelsAsFrames(open(docked_sdf_path).read())
        v.setStyle(
            {"model": -3},
            {"stick": {"colorscheme": "greenCarbon", "radius": stick_radius * 1.25}},
        )

    # Set view
    if zoom_ligand and ligand_path:
        v.zoomTo({"model": -2})
    else:
        v.zoomTo()

    v.rotate(rotate_degree)

    # Set animation if multiple models are present
    if ligand_path or docked_sdf_path:
        v.animate({"interval": animation_interval})

    return v


# Example usage:
# view = view_3d('receptor.pdb', 'ligand.pdb', 'docked.pdb',
#                receptor_highlight=100, style='cartoon', surface=True,
#                surface_opacity=0.5, surface_color='skyblue')
# view.show()


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
    line_color="black",
    marker_size=15,
    leaf_color="CNN_score",
    render_mode="svg",
    leaf_y=0,
    leaf_color_discrete_map=None,
    leaf_category_orders=None,
    template="simple_white",
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
                hover_name="Mode",
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
    pose_order = leaf_data["Mode"].to_numpy() - 1

    fig = go.Figure(
        data=go.Heatmap(
            z=df_rmsd,
            x=np.arange(len(pose_order)),
            y=np.arange(len(pose_order)),
            colorscale="Viridis",
            showscale=False,
        )
    )

    return fig


def _concat_subplots(
    figures,
    width,
    height,
):
    from plotly.subplots import make_subplots  # type: ignore

    # make subplots
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05, shared_xaxes=True)

    for i, figure in enumerate(figures):
        if isinstance(figure, go.Figure):
            # This is a figure, access the traces within it.
            for trace in range(len(figure["data"])):
                fig.append_trace(figure["data"][trace], row=i + 1, col=1)
        else:
            # Assume this is a trace, add directly.
            fig.append_trace(figure, row=i + 1, col=1)

    fig.update_xaxes(visible=False)
    fig.update_layout(
        width=width,
        height=height,
        hovermode="closest",
        plot_bgcolor="white",
    )
    fig.update(layout_coloraxis_showscale=False)

    return fig
