#!/usr/bin/env python3
"""Download artifacts, then start the viewer."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
download = subprocess.run([sys.executable, str(root / "scripts" / "download_data.py")], cwd=root)
if download.returncode:
    raise SystemExit(download.returncode)
indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-index"], cwd=root)
if indexed.returncode:
    raise SystemExit(indexed.returncode)
raise SystemExit(subprocess.call([sys.executable, str(root / "scripts" / "run_web.py")], cwd=root))
