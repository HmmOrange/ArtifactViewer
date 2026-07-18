#!/usr/bin/env python3
"""JSON API for evaluation artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_PATH = DATA / ".artifact-viewer-index.json"
WEB_INDEX_PATH = ROOT / "web" / "src" / "qa-index.json"
INDEX_VERSION = 3
QUESTION_KEYS = ("question", "query", "prompt", "qa", "input")
GOLD_KEYS = ("gold", "golden", "answer", "reference", "ground_truth", "target", "label")
PRED_KEYS = ("prediction", "predicted", "pred", "response", "output", "model_answer", "model_output")
csv.field_size_limit(10 * 1024 * 1024)


def value(record: dict, keys: tuple[str, ...]):
    lowered = {str(key).lower().replace("-", "_"): item for key, item in record.items()}
    return next((lowered[key] for key in keys if key in lowered), None)


def flatten(item):
    if isinstance(item, list):
        for child in item:
            yield from flatten(child)
    elif isinstance(item, dict):
        rows = next((item[key] for key in ("records", "results", "examples", "items", "data") if key in item), None)
        if isinstance(rows, list):
            yield from flatten(rows)
        else:
            yield item


def answers_match(gold, prediction) -> bool:
    left, right = str(gold).strip().casefold(), str(prediction).strip().casefold()
    if left == right:
        return True
    number = r"[-+]?\d+(?:\.\d+)?"
    left_numbers, right_numbers = re.findall(number, left), re.findall(number, right)
    return len(left_numbers) == len(right_numbers) == 1 and float(left_numbers[0]) == float(right_numbers[0])


def read_file(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def load_records(pipeline_filter: str = "", dataset_filter: str = "") -> list[dict]:
    records = []
    search_root = DATA
    if pipeline_filter and dataset_filter:
        search_root = DATA / "Artifacts" / pipeline_filter / dataset_filter
    if not search_root.exists():
        return records
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        try:
            for raw in flatten(read_file(path)):
                if not isinstance(raw, dict):
                    continue
                question = value(raw, QUESTION_KEYS)
                gold = value(raw, GOLD_KEYS)
                prediction = value(raw, PRED_KEYS)
                missing_output = path.name.lower() == "input.jsonl" and not (path.parent / "output.jsonl").exists()
                failed = raw.get("error") or str(raw.get("status", "")).lower() in {"error", "failed", "failure"}
                if question is None or (prediction is None and not missing_output and not failed):
                    continue

                parts = path.relative_to(DATA).parts
                pipeline = pipeline_filter or (parts[1] if len(parts) > 1 else "Unknown")
                dataset = dataset_filter or (parts[2] if len(parts) > 2 else "Unknown")
                explicit_correct = value(raw, ("correct", "is_correct", "exact_match", "em"))
                if missing_output or failed or prediction is None:
                    status = "error"
                elif explicit_correct is not None:
                    status = "correct" if explicit_correct in {True, 1, "1", "true", "True"} else "wrong"
                else:
                    status = "correct" if gold is not None and answers_match(gold, prediction) else "wrong"

                source = str(path.relative_to(DATA))
                records.append({
                    "id": f"{source}:{len(records)}",
                    "pipeline": pipeline,
                    "dataset": dataset,
                    "question": question,
                    "gold": gold,
                    "prediction": prediction,
                    "status": status,
                    "source": source,
                })
        except (csv.Error, OSError, ValueError, UnicodeDecodeError):
            continue
    return records


def load_catalog() -> dict[str, list[str]]:
    artifacts = DATA / "Artifacts"
    if not artifacts.exists():
        return {}
    return {
        pipeline.name: sorted(child.name for child in pipeline.iterdir() if child.is_dir())
        for pipeline in sorted(artifacts.iterdir())
        if pipeline.is_dir()
    }


def default_selection(catalog: dict[str, list[str]]) -> dict[str, str]:
    artifacts = DATA / "Artifacts"
    for pipeline, datasets in catalog.items():
        for dataset in datasets:
            dataset_path = artifacts / pipeline / dataset
            has_runs = (dataset_path / "runs").is_dir() or any(dataset_path.glob("*/runs"))
            if has_runs:
                return {"pipeline": pipeline, "dataset": dataset}
    first_pipeline = next(iter(catalog), "")
    first_dataset = catalog.get(first_pipeline, [""])[0] if first_pipeline else ""
    return {"pipeline": first_pipeline, "dataset": first_dataset}


@lru_cache(maxsize=4)
def load_common_questions(dataset: str) -> list[dict]:
    questions = []
    root = DATA / "Artifacts" / "ST-raptor" / dataset
    for path in root.rglob("input.jsonl"):
        try:
            for raw in flatten(read_file(path)):
                if not isinstance(raw, dict):
                    continue
                question = value(raw, QUESTION_KEYS)
                if question is None:
                    continue
                questions.append({
                    "key": str(raw.get("id") or path.parent.name),
                    "question": question,
                    "gold": value(raw, GOLD_KEYS),
                    "source": str(path.relative_to(DATA)),
                })
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    return questions


def missing_prediction_records(pipeline: str, dataset: str) -> list[dict]:
    return [{
        "id": f"{pipeline}:{dataset}:{item['key']}",
        "pipeline": pipeline,
        "dataset": dataset,
        "question": item["question"],
        "gold": item["gold"],
        "prediction": None,
        "status": "error",
        "source": f"{item['source']} (no {pipeline} prediction artifact)",
    } for item in load_common_questions(dataset)]


def load_output_records(pipeline: str, dataset: str) -> list[dict]:
    output_dir = DATA / "Outputs" / pipeline / dataset
    reports = list(output_dir.glob("report_*.json"))
    if not reports:
        return []

    def report_number(path: Path) -> int:
        suffix = path.stem.removeprefix("report_")
        return int(suffix) if suffix.isdigit() else 0

    report_path = max(reports, key=report_number)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    records = []
    for index, raw in enumerate(payload.get("results", [])):
        error = raw.get("error")
        exact_match = raw.get("exact_match")
        if error:
            status = "error"
        else:
            try:
                status = "correct" if float(exact_match or 0) >= 1 else "wrong"
            except (TypeError, ValueError):
                status = "wrong"
        records.append({
            "id": f"{pipeline}:{dataset}:{raw.get('sample_id') or index}",
            "pipeline": pipeline,
            "dataset": dataset,
            "question": raw.get("question") or "",
            "gold": raw.get("gold_answer"),
            "prediction": raw.get("predicted_answer"),
            "status": status,
            "source": str(report_path.relative_to(DATA)),
        })
    return records


def build_index() -> dict:
    indexed = {}
    for pipeline, datasets in load_catalog().items():
        indexed[pipeline] = {}
        for dataset in datasets:
            print(f"Indexing {pipeline} / {dataset}...", flush=True)
            indexed[pipeline][dataset] = load_output_records(pipeline, dataset)
    payload = {"version": INDEX_VERSION, "records": indexed}
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = INDEX_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(INDEX_PATH)
    WEB_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


@lru_cache(maxsize=1)
def load_index() -> dict:
    if INDEX_PATH.exists():
        try:
            payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if payload.get("version") == INDEX_VERSION:
                return payload
        except (OSError, ValueError):
            pass
    return build_index()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/api/health":
            payload = {"ok": True}
        elif request.path == "/api/catalog":
            catalog = load_catalog()
            payload = {
                "catalog": catalog,
                "default": default_selection(catalog),
            }
        elif request.path == "/api/records":
            params = parse_qs(request.query)
            pipeline = params.get("pipeline", [""])[0]
            dataset = params.get("dataset", [""])[0]
            records = load_index().get("records", {}).get(pipeline, {}).get(dataset, [])
            query = params.get("q", [""])[0].strip().casefold()
            try:
                offset = max(0, int(params.get("offset", ["0"])[0]))
                limit = min(100, max(1, int(params.get("limit", ["40"])[0])))
            except ValueError:
                self.send_error(400, "offset and limit must be integers")
                return
            matches = [
                record for record in records
                if record["pipeline"] == pipeline
                and record["dataset"] == dataset
                and (not query or query in str(record["question"]).casefold())
            ]
            totals = Counter(record["status"] for record in records)
            payload = {
                "records": matches[offset:offset + limit],
                "total": len(matches),
                "offset": offset,
                "totals": {status: totals.get(status, 0) for status in ("correct", "wrong", "error")},
            }
        else:
            self.send_error(404)
            return
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        return


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the artifact viewer API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if args.build_index:
        INDEX_PATH.unlink(missing_ok=True)
        load_records.cache_clear()
        load_index.cache_clear()
        build_index()
        return
    print("Preparing QA index...", flush=True)
    load_index()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.ready_file:
        args.ready_file.write_text(str(server.server_port), encoding="ascii")
    print(f"Artifact API: http://{args.host}:{args.port}/api/catalog", flush=True)
    try:
        server.serve_forever()
    finally:
        if args.ready_file:
            args.ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
