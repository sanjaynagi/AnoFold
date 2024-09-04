import plotly.graph_objects as go
import numpy as np
import pandas as pd
import py3Dmol

def _view_3d(self, receptor_highlight=None, sticks=False):
    v = py3Dmol.view()
    v.addModel(open(self.receptor_path).read())
    if sticks:
        v.setStyle({'cartoon':{},'stick':{'radius':.1}})
    else:
        v.setStyle({'cartoon':{}})
    if receptor_highlight:
        for i in range(receptor_highlight-3, receptor_highlight+3):
            v.setStyle({'model': -1, 'serial': i}, {"cartoon": {'color': 'yellow'}, 'stick':{'radius':.3, 'color':'yellow'}})
    v.addModel(open(self.ligand_path).read())
    v.setStyle({'model':1},{'stick':{'colorscheme':'dimgrayCarbon','radius':.125}})
    v.addModelsAsFrames(open(self.docked_path).read())
    v.setStyle({'model':2},{'stick':{'colorscheme':'greenCarbon'}})
    v.zoomTo({'model':1})
    v.rotate(90)
    v.animate({'interval':5000})
    return v

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