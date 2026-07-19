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

Download the artifact archive from Google Drive:

[Download the data archive](https://drive.google.com/file/d/1nn50mdHGA2QrVRqjo8HM0THHBGrHehvf/view?usp=sharing)

Open the link and select **Download** in Google Drive. If Google Drive displays a warning that it cannot scan the large file for viruses, select **Download anyway**.

## 2. Unzip and place the data

1. Locate the downloaded ZIP file, usually in the `Downloads` folder.
2. Extract or unzip the archive.
3. Open the extracted folder and find the folder that directly contains `Artifacts`, `Datasets`, `Log_Tabular_Models`, `Log_tabAgent_Siflex`, and `Outputs`.
4. Rename that folder to `data`, using lowercase letters, if it has a different name.
5. Move `data` into the `ArtifactViewer` repository root, beside `api`, `pipeline`, `scripts`, and `web`.

The expected repository structure is:

```text
ArtifactViewer/
|-- api/
|   |-- __init__.py
|   `-- server.py
|-- pipeline/
|-- scripts/
|   |-- download_and_run.py
|   |-- download_data.py
|   `-- run_web.py
|-- web/
|   |-- src/
|   |-- index.html
|   |-- package-lock.json
|   |-- package.json
|   `-- vite.config.js
|-- README.md
`-- data/
    |-- Artifacts/
    |-- Datasets/
    |-- Log_Tabular_Models/
    |-- Log_tabAgent_Siflex/
    `-- Outputs/
```

The five artifact folders must be directly inside `data`. Do not keep an extra extracted-folder level.

Incorrect:

```text
ArtifactViewer/data/extracted-folder/Artifacts/
```

Correct:

```text
ArtifactViewer/data/Artifacts/
```

On macOS or Linux, the final move can also be done from a terminal. Replace the example source path with the actual extracted folder:

```bash
cd /path/to/ArtifactViewer
mv "/path/to/extracted-folder" data
```

On Windows, use File Explorer to extract the ZIP, rename the artifact folder to `data`, and move it beside `api`, `pipeline`, `scripts`, and `web`.

## 3. Create a Python environment

Creating a virtual environment is recommended.

### macOS or Linux

```bash
cd /path/to/ArtifactViewer
    python3 -m venv .venv
source .venv/bin/activatxe
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

Run the launcher from the repository root. The current directory must be the folder that contains `api`, `scripts`, `web`, and `data`.

Important directory rule:

- Run `python scripts/run_web.py` from `ArtifactViewer`.
- Run `npm install` or `npm run dev` only from `ArtifactViewer/web`.
- Do not run `npm install` directly from `ArtifactViewer`; there is no `package.json` in the repository root.

For the normal setup, you do not need to run `npm install` manually. The Python launcher enters `web` and installs the frontend dependencies when needed.

### macOS or Linux

```bash
python scripts/run_web.py
```

### Windows

```powershell
cd C:\path\to\ArtifactViewer
python scripts\run_web.py
```

Before pressing Enter, the command prompt should end with `ArtifactViewer>`, not `ArtifactViewer\web>`:

```text
C:\path\to\ArtifactViewer>python scripts\run_web.py
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

The `cd web` step is required because [web/package.json](web/package.json) contains the frontend dependencies.

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

### `npm ERR! ENOENT: Could not read package.json`

This error means `npm install` was run from the repository root:

```text
C:\path\to\ArtifactViewer>npm install
```

That location does not contain `package.json`. Move into `web` before running npm:

```bat
cd /d C:\path\to\ArtifactViewer\web
npm install
cd ..
python scripts\run_web.py
```

The expected npm location is:

```text
C:\path\to\ArtifactViewer\web>npm install
```

### Vite reports `Failed to resolve import "react-markdown"`

The frontend dependencies are missing or only partially installed. Stop the running viewer with `Ctrl+C`, then reinstall them inside `web`.

Windows Command Prompt:

```bat
cd /d C:\path\to\ArtifactViewer\web
npm install
cd ..
python scripts\run_web.py
```

If the same error remains, perform a clean installation:

```bat
cd /d C:\path\to\ArtifactViewer\web
rmdir /s /q node_modules
npm ci
cd ..
python scripts\run_web.py
```

Windows PowerShell clean installation:

```powershell
cd C:\path\to\ArtifactViewer\web
Remove-Item -Recurse -Force node_modules
npm ci
cd ..
python scripts\run_web.py
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
