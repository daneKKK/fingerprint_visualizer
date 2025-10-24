# Interactive Atomic Configuration Visualizer

This Dash application allows for the visualization of multi-component vector representations of atomic configurations in three-dimensional space using the UMAP algorithm, accompanying each point on the plot with a visual representation rendered via OVITO.

## Key Features

- **Interactive 3D Plot:** Displays a UMAP projection with clustering, where each point represents an atomic structure.
- **Click-to-Render:** When a point on the plot is clicked, a high-quality image of the corresponding structure appears on the right.
- **OVITO Engine:** The OVITO visualization tool is used for image generation, ensuring image quality and a clear demonstration of the results.
- **Asynchronous Tasks:** Rendering is performed in the background without blocking or "freezing" the application interface. A loading indicator is displayed during rendering.
- **Automatic Launch:** When the server starts, the application automatically opens in a new browser tab.

## Setup and Launch

### 1. Requirements
- Python >= 3.9
- A virtual environment is recommended

### 2. Installation
1.  **Clone or download the archive** with the project and unzip it.
2.  **Create and activate a virtual environment:**
    ```bash
    # Create an environment in the 'venv' folder
    python -m venv venv

    # Activate it
    # For Windows:
    venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install all necessary dependencies** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Launching the application
To start the server, execute the following command in the terminal while in the project's root folder:
```bash
python app.py
```
After launching, you will see initialization and pre-rendering messages in the console. After a few seconds, a tab with the application will automatically open in your default browser at `http://127.0.0.1:8050`.

## How to use the application
- **Left Panel:** The 3D UMAP plot. It can be rotated and scaled using the mouse. The points are colored according to their clusters.
- **Right Panel:** The visualization area.
- **Action:** **Click** the left mouse button on any point on the left plot.
- **Waiting:** A loading indicator will appear in place of the image on the right. This means that OVITO is generating the image in the background. The process may take a few seconds.
- **Result:** After the rendering is complete, the loading indicator will disappear, and a new image of the selected atomic structure with bonds will appear.