# Artifact Viewer

Artifact Viewer is a local web UI for browsing pipeline artifacts and comparing the full evaluation flow:

```text
q + X -> Z -> W -> Y vs Y*
```

The viewer currently includes GraphOtter, SpreadsheetAgent, ST-Raptor, TableAgent/SIFLEX, and the uploaded Data Lake tabular-model runs.

## Prerequisites

Install these tools before running the repository:

- Python 3.10 or newer
- Node.js 18 or newer, including `npm`
- Enough local disk space for the downloaded artifact bundle

Check the installed versions:

```bash
python3 --version
node --version
npm --version
```

On Windows, use `py --version` if the `python` command is unavailable.

## 1. Download the artifact data

Download the Google Drive folder named `[Logs] All pipelines`:

https://drive.google.com/drive/u/0/folders/1kSXiNRfyiZAEsqtKzStyeO1JhhtWhQMt

Google Drive may download the folder as a ZIP archive. Extract it before continuing.

## 2. Rename and place the folder

Rename the extracted `[Logs] All pipelines` folder to `data`, using lowercase letters, and place it in the repository root.

The resulting layout must look like this:

```text
ArtifactViewer/
|-- api/
|-- scripts/
|-- web/
|-- README.md
`-- data/
    |-- Artifacts/
    |-- Datasets/
    |-- Log_Tabular_Models/
    |-- Log_tabAgent_Siflex/
    `-- Outputs/
```

Do not place the downloaded folder one level too deep. This is incorrect:

```text
ArtifactViewer/data/[Logs] All pipelines/Artifacts/
```

The correct path is:

```text
ArtifactViewer/data/Artifacts/
```

On macOS or Linux, you can move and rename the extracted folder from a terminal:

```bash
cd /path/to/ArtifactViewer
mv "/path/to/[Logs] All pipelines" data
```

On Windows, rename the folder in File Explorer to `data`, then move it beside `api`, `scripts`, and `web`.

## 3. Create a Python environment

Creating a virtual environment is recommended.

### macOS or Linux

```bash
cd /path/to/ArtifactViewer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy openpyxl pyyaml
```

### Windows PowerShell

```powershell
cd C:\path\to\ArtifactViewer
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy openpyxl pyyaml
```

If PowerShell blocks virtual-environment activation, run this command once in the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run `.venv\Scripts\Activate.ps1` again.

## 4. Run the web viewer

Run the launcher from the repository root.

### macOS or Linux

```bash
python scripts/run_web.py
```

### Windows

```powershell
python scripts\run_web.py
```

The launcher automatically:

1. Installs the frontend packages with `npm install` when `web/node_modules` is missing.
2. Builds or refreshes the artifact indexes when required.
3. Starts the Python artifact API at `http://127.0.0.1:8766`.
4. Starts the Vite web UI at `http://127.0.0.1:8765`.

Open this address in a browser:

http://127.0.0.1:8765

Keep the terminal running while using the viewer. Press `Ctrl+C` in that terminal to stop both services.

The first run can take longer because npm dependencies may need to be installed and the data indexes may need to be generated.

## Rebuild the indexes

Rebuild all indexes after replacing or changing files inside `data`:

```bash
python -m api.server --build-index
python scripts/run_web.py
```

On Windows, the equivalent commands are:

```powershell
python -m api.server --build-index
python scripts\run_web.py
```

## Run the frontend and API separately

This is useful when debugging.

Terminal 1, from the repository root:

```bash
python -m api.server
```

Terminal 2:

```bash
cd web
npm install
npm run dev -- --host 127.0.0.1
```

Then open `http://127.0.0.1:8765`.

## Troubleshooting

### The page is blank or cannot load artifacts

- Check that the terminal running `scripts/run_web.py` is still open and has no error.
- Confirm that both `http://127.0.0.1:8765` and `http://127.0.0.1:8766/api/health` respond.
- Hard-refresh the browser after restarting the launcher.
- Rebuild the indexes if the contents of `data` changed.

### `data` cannot be found

Confirm that the folder is named exactly `data` and is located in the repository root:

```text
ArtifactViewer/data/Datasets/
```

### `npm` or `node` cannot be found

Install Node.js 18 or newer, close and reopen the terminal, then verify:

```bash
node --version
npm --version
```

### A Python module is missing

Activate the virtual environment and reinstall the required packages:

```bash
python -m pip install numpy openpyxl pyyaml
```

### Port 8765 or 8766 is already in use

Stop the older Artifact Viewer terminal or process, then run `scripts/run_web.py` again. Only one viewer instance should use these ports at a time.

### macOS cannot reveal a file in Finder

The viewer still works if Finder access is unavailable. The recorded artifact path remains visible in the UI and can be opened manually from the repository's `data` folder.
