#!/usr/bin/env python3
"""Run the Vite frontend and artifact API together."""
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    web = root / "web"
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not npm:
        print("Node.js/npm is required to run the web viewer.", file=sys.stderr)
        return 2
    if not (web / "node_modules").exists():
        installed = subprocess.run([npm, "install"], cwd=web)
        if installed.returncode:
            return installed.returncode

    index_path = web / "src" / "qa-index.json"
    if not index_path.exists():
        indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-index"], cwd=root)
        if indexed.returncode:
            return indexed.returncode

    api = subprocess.Popen([sys.executable, "-m", "api.server"], cwd=root)
    try:
        return subprocess.call([npm, "run", "dev", "--", "--host", "127.0.0.1"], cwd=web)
    finally:
        api.terminate()
        api.wait()


if __name__ == "__main__":
    raise SystemExit(main())
