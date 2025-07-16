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

# --- Часть 1: Код, выполняемый каждым процессом (основным и рабочим) ---

# Импорт тяжелых библиотек
try:
    from ovito.io import ase as ovito_ase
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport, ParticlesVis
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from ovito.modifiers import CreateBondsModifier, DeleteSelectedModifier
    from ovito.vis import BondsVis
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    exit()

# Константы, доступные всем процессам
XYZ_FILENAME = "dataset.xyz"
EMBEDDING_FILENAME = "3d_embedding.npy"
LABELS_FILENAME = "labels.npy"
TRIM_FACTOR = 5

# Настройка менеджера фоновых задач
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# Загрузка данных, необходимых для графика UMAP. Они маленькие и нужны всем.
print(f"INFO: Процесс {os.getpid()} загружает данные для UMAP...")
try:
    embedding_3d = np.load(EMBEDDING_FILENAME)
    labels = np.load(LABELS_FILENAME)
except FileNotFoundError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не найден файл {e.filename}.")
    exit()

df = pd.DataFrame({
    'UMAP1': embedding_3d[:, 0], 'UMAP2': embedding_3d[:, 1], 'UMAP3': embedding_3d[:, 2],
    'cluster': labels.astype(str), 'index': range(len(labels))
})

# Создание фигуры UMAP. Это быстро и нужно всем.
fig_3d = px.scatter_3d(
    df, x="UMAP1", y="UMAP2", z="UMAP3", color="cluster", custom_data=['index'],
    title="3D UMAP проекция с кластеризацией",
    labels={'UMAP1': 'UMAP-1', 'UMAP2': 'UMAP-2', 'UMAP3': 'UMAP-3'}, opacity=0.8,
)
fig_3d.update_traces(marker=dict(size=3.5, symbol="diamond"), hovertemplate="<b>Точка: %{customdata[0]}</b><extra></extra>")
fig_3d.update_layout(legend_title_text='Кластер', template="plotly_dark")
print(f"INFO: Процесс {os.getpid()} подготовил UMAP.")


# Определение функции рендеринга
def render_with_ovito(atoms_obj):
    data_collection = ovito_ase.ase_to_ovito(atoms_obj)
    pipeline = Pipeline(source=StaticSource(data=data_collection))
    pipeline.source.data.particles.vis.shape = ParticlesVis.Shape.Sphere
    pipeline.source.data.particles.vis.radius = 0.5
    pipeline.modifiers.append(CreateBondsModifier(mode = CreateBondsModifier.Mode.VdWRadius))
    #pipeline.source.data.bonds.vis.enabled = True
    #pipeline.source.data.bonds.vis.width = 0.15 # Задаем толщину связей
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

# Инициализация объекта Dash приложения
app = dash.Dash(__name__)

# Определение фонового callback
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
    
    # Вычисляем реальный индекс в большом xyz файле
    full_xyz_index = point_index * TRIM_FACTOR
    
    title = f"Конфигурация для точки {point_index} (структура {full_xyz_index} в датасете)"
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Целевое чтение одного кадра из файла ---
    # Рабочий процесс не держит в памяти весь датасет!
    print(f"INFO: Рабочий процесс {os.getpid()} читает кадр {full_xyz_index} из {XYZ_FILENAME}...")
    selected_atoms = ase.io.read(XYZ_FILENAME, index=full_xyz_index)
    
    image_src = render_with_ovito(selected_atoms)
    
    return title, image_src


# --- Часть 2: Код, выполняемый ТОЛЬКО основным процессом при запуске ---

if __name__ == '__main__':
    # Эта часть кода НЕ будет выполняться рабочими процессами
    
    print("\n--- ЗАПУСК ОСНОВНОГО СЕРВЕРА ---")
    
    # Зададим хост и порт как переменные для удобства
    HOST = '127.0.0.1'
    PORT = 8050
    
    # Предварительный рендеринг начального изображения
    print("INFO: Основной сервер выполняет предварительный рендеринг...")
    initial_atoms = ase.io.read(XYZ_FILENAME, index=0)
    initial_image_src = render_with_ovito(initial_atoms)
    initial_title = "Кликните по точке для визуализации. Показана структура 0."
    print("INFO: Предварительный рендеринг завершен.")
    
    # Создание и назначение верстки (layout)
    app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#DDDDDD', 'fontFamily': 'sans-serif'}, children=[
        html.H1("Интерактивная визуализация конфигураций", style={'textAlign': 'center', 'padding': '20px'}),
        html.Div(className='row', style={'display': 'flex', 'padding': '20px'}, children=[
            html.Div(className='six columns', style={'width': '60%'}, children=[
                dcc.Graph(id='3d-scatter-plot', figure=fig_3d, style={'height': '80vh'})
            ]),
            html.Div(className='six columns', style={'width': '40%', 'textAlign': 'center', 'paddingTop': '50px'}, children=[
                html.H4(id='point-info', children=initial_title, style={'height': '50px'}),
                dcc.Loading(id="loading-spinner", type="circle",
                            children=html.Img(id='atom-image', src=initial_image_src, style={'maxWidth': '100%', 'maxHeight': '60vh', 'border': '1px solid #555', 'borderRadius': '10px'}))
            ])
        ])
    ])
    
    # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
    
    # 1. Определяем функцию, которая откроет браузер
    def open_browser():
        webbrowser.open_new(f"http://{HOST}:{PORT}")

    # 2. Запускаем эту функцию с задержкой в 2 секунды в отдельном потоке.
    #    Это гарантирует, что сервер успеет запуститься.
    Timer(2, open_browser).start()
    
    # 3. Запускаем сам сервер
    print(f"INFO: Сервер готов к приему подключений на http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)
    
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
