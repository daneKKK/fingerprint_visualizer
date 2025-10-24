import dash
from dash import dcc, html, callback, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import ase.io
import io
import base64
import diskcache
from dash import DiskcacheManager
import os
import webbrowser
from threading import Timer

try:
    from ovito.io import ase as ovito_ase
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport, ParticlesVis
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from ovito.modifiers import ComputePropertyModifier
    from ovito.vis import BondsVis
except ImportError as e:
    print(f"Import error: {e}")
    exit()

XYZ_FILENAME = "./heb2/dataset.xyz"
EMBEDDING_FILENAME = "./heb2/3d_embedding.npy"
INDICES_FILENAME = "./heb2/indices.npy"
LABELS_FILENAME = "./heb2/labels.npy"
TRIM_FACTOR = 100

XYZ_TEST_FILENAME = "./heb2/test.xyz"
EMBEDDING_TEST_FILENAME = "./heb2/3d_embedding_test.npy"
INDICES_TEST_FILENAME = "./heb2/indices_test.npy"
LABELS_TEST_FILENAME = "./heb2/labels_test.npy"

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

print(f"INFO: Process {os.getpid()} is loading data for UMAP...")
try:
    embedding_3d = np.load(EMBEDDING_FILENAME)
    labels = np.load(LABELS_FILENAME)
    indices = np.load(INDICES_FILENAME).astype(int)
    train_size = len(embedding_3d)
    
    embedding_3d_test = np.load(EMBEDDING_TEST_FILENAME)
    labels_test = np.load(LABELS_TEST_FILENAME)
    indices_test = np.load(INDICES_TEST_FILENAME).astype(int)
    
    embedding_3d = np.concatenate([embedding_3d, embedding_3d_test], axis=0)
    labels = np.concatenate([labels, labels_test])
    indices = np.concatenate([indices, indices_test])
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: File not found {e.filename}.")
    exit()

print(embedding_3d.shape, labels.shape)

df = pd.DataFrame({
    'UMAP1': embedding_3d[:, 0], 'UMAP2': embedding_3d[:, 1], 'UMAP3': embedding_3d[:, 2],
    'cluster': labels.astype(str), 'index': range(len(labels))
})

fig_3d = px.scatter_3d(
    df, x="UMAP1", y="UMAP2", z="UMAP3", color="cluster", custom_data=['index'],
    title="3D dataset projection",
    labels={'UMAP1': 'X', 'UMAP2': 'Y', 'UMAP3': 'Z'}, opacity=0.8,
)
fig_3d.update_traces(marker=dict(size=3.5, symbol="diamond"), hovertemplate="<b>Point: %{customdata[0]}</b><extra></extra>")
fig_3d.update_layout(legend_title_text='Cluster', template="plotly_white")
print(f"INFO: Process {os.getpid()} has prepared the UMAP.")

def render_with_ovito(atoms_obj, atom_index=-1, labels=None):
    data_collection = ovito_ase.ase_to_ovito(atoms_obj)
    pipeline = Pipeline(source=StaticSource(data=data_collection))
    pipeline.source.data.particles.vis.shape = ParticlesVis.Shape.Sphere
    pipeline.source.data.particles.vis.radius = 0.5
    pipeline.modifiers.append(ComputePropertyModifier(
        output_property="Radius",
        expressions=["ParticleType == \"B\" ? 0.85 : 1.5"]
    ))
    pipeline.source.data.particles.identifiers_ = np.arange(len(atoms_obj))
    print(pipeline.source.data.particles.identifiers)
    pipeline.modifiers.append(ComputePropertyModifier(
        output_property="Transparency",
        expressions=[f"ParticleIdentifier == {atom_index}? 0.0 : 0.7"]
    ))
    pipeline.add_to_scene()
    vp = Viewport(type=Viewport.Type.PERSPECTIVE, camera_dir=(-1, -1, -1), fov=np.deg2rad(40.0))
    vp.zoom_all(size=(800, 800))
    q_image = vp.render_image(size=(800, 800), alpha=True)
    pipeline.remove_from_scene()
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    q_image.save(buffer, "PNG")
    encoded_image = base64.b64encode(bytes(byte_array.data())).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

app = dash.Dash(__name__)

@callback(
    output=[Output('point-info', 'children'), Output('atom-image', 'src')],
    inputs=Input('3d-scatter-plot', 'clickData'),
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True
)
def update_image_on_click(clickData):
    if clickData is None:
        raise PreventUpdate

    point_info = clickData['points'][0]
    curve_number = point_info['curveNumber']
    point_number = point_info['pointNumber']
    point_index = fig_3d.data[curve_number].customdata[point_number][0]
    
    full_xyz_index = point_index
    
    
    if full_xyz_index < train_size:
        selected_atoms = ase.io.read(XYZ_FILENAME, index=indices[full_xyz_index])
        atom_index = point_index - np.min(np.nonzero(indices == indices[full_xyz_index]))
        print(point_index, indices[full_xyz_index], np.min(np.nonzero(indices == indices[full_xyz_index])), atom_index)
        chosen_labels = labels[indices == indices[full_xyz_index]]
        title = f"Configuration for point {point_index} (structure {indices[full_xyz_index]} in the dataset)"
    
        print(f"INFO: Worker process {os.getpid()} is reading frame {full_xyz_index} from {XYZ_FILENAME}...")
    else:
        selected_atoms = ase.io.read(XYZ_TEST_FILENAME, index=indices[full_xyz_index])
        chosen_labels = labels_test[indices_test == indices_test[full_xyz_index-train_size]]
        atom_index = full_xyz_index - train_size - indices_test[full_xyz_index-train_size]
        title = f"Configuration for point {point_index} (structure {indices_test[full_xyz_index-train_size]} in the  test dataset)"
    
        print(f"INFO: Worker process {os.getpid()} is reading frame {indices_test[full_xyz_index-train_size]} from {XYZ_TEST_FILENAME}...")
    
    image_src = render_with_ovito(selected_atoms, atom_index, chosen_labels)
    
    return title, image_src

if __name__ == '__main__':
    print("\n--- STARTING MAIN SERVER ---")
    
    HOST = '127.0.0.1'
    PORT = 8050
    
    print("INFO: Main server is performing pre-rendering...")
    initial_atoms = ase.io.read(XYZ_FILENAME, index=0)
    initial_image_src = render_with_ovito(initial_atoms)
    initial_title = "Click on a point to visualize. Structure 0 is shown."
    print("INFO: Pre-rendering complete.")
    
    app.layout = html.Div(style={'backgroundColor': '#ffffff', 'color': '#DDDDDD', 'fontFamily': 'sans-serif'}, children=[
        html.H1("Interactive Visualization of Configurations", style={'textAlign': 'center', 'padding': '20px', 'color': '#111111'}),
        html.Div(className='row', style={'display': 'flex', 'padding': '20px'}, children=[
            html.Div(className='six columns', style={'width': '60%'}, children=[
                dcc.Graph(id='3d-scatter-plot', figure=fig_3d, style={'height': '80vh'})
            ]),
            html.Div(className='six columns', style={'width': '40%', 'textAlign': 'center', 'paddingTop': '50px'}, children=[
                html.H4(id='point-info', children=initial_title, style={'height': '50px', 'color': '#111111'}),
                dcc.Loading(id="loading-spinner", type="circle",
                            children=html.Img(id='atom-image', src=initial_image_src, style={'maxWidth': '100%', 'maxHeight': '60vh', 'border': '1px solid #555', 'borderRadius': '10px'}))
            ])
        ])
    ])
    
    def open_browser():
        webbrowser.open_new(f"http://{HOST}:{PORT}")

    Timer(2, open_browser).start()
    
    print(f"INFO: Server is ready to accept connections at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)