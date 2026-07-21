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
    siflex_index_path = web / "src" / "siflex-index.json"
    tabular_models_index_path = web / "src" / "tabular-models-index.json"
    mismatch_index_path = web / "src" / "mismatch-index.json"
    failure_analysis_index_path = web / "src" / "failure-analysis-index.json"
    failure_analysis_sources = (
        root / "analysis" / "failure_modes" / "summary.csv",
        root / "analysis" / "failure_modes" / "all_failed_cases.csv",
    )
    failure_cases_path = root / "analysis" / "failure_modes" / "all_failed_cases.csv"
    output_roots = (root / "data" / "Outputs", root / "data" / "Outputs_llm")
    evaluation_reports = [
        report
        for output_root in output_roots if output_root.is_dir()
        for report in output_root.rglob("report_*.json")
    ]
    failure_analysis_needs_refresh = (
        not failure_cases_path.is_file()
        or any(report.stat().st_mtime_ns > failure_cases_path.stat().st_mtime_ns for report in evaluation_reports)
    )
    if failure_analysis_needs_refresh:
        analyzed = subprocess.run([sys.executable, "scripts/analyze_failure_modes.py"], cwd=root)
        if analyzed.returncode:
            return analyzed.returncode
    golden_cases_path = root / "data" / "Datasets" / "SiFlex" / "golden_tests" / "compiled" / "golden_cases.json"
    siflex_needs_refresh = (
        golden_cases_path.is_file()
        and (
            not siflex_index_path.exists()
            or golden_cases_path.stat().st_mtime_ns > siflex_index_path.stat().st_mtime_ns
        )
    )
    benchmark_needs_refresh = index_path.exists() and any(
        report.stat().st_mtime_ns > index_path.stat().st_mtime_ns
        for report in evaluation_reports
    )
    failure_analysis_needs_refresh = any(
        source.is_file() and (not failure_analysis_index_path.exists() or source.stat().st_mtime_ns > failure_analysis_index_path.stat().st_mtime_ns)
        for source in failure_analysis_sources
    )
    if (
        not index_path.exists()
        or not siflex_index_path.exists()
        or not tabular_models_index_path.exists()
        or not mismatch_index_path.exists()
        or not failure_analysis_index_path.exists()
        or benchmark_needs_refresh
    ):
        indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-index"], cwd=root)
        if indexed.returncode:
            return indexed.returncode
    else:
        if siflex_needs_refresh:
            indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-siflex-index"], cwd=root)
            if indexed.returncode:
                return indexed.returncode
        mismatch_sources = (index_path, siflex_index_path)
        if siflex_needs_refresh or any(path.stat().st_mtime_ns > mismatch_index_path.stat().st_mtime_ns for path in mismatch_sources):
            indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-mismatch-index"], cwd=root)
            if indexed.returncode:
                return indexed.returncode
        if failure_analysis_needs_refresh:
            indexed = subprocess.run([sys.executable, "-m", "api.server", "--build-failure-analysis-index"], cwd=root)
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
