#!/usr/bin/env python3
"""Evaluate saved benchmark answers with an OpenAI-compatible LLM judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "llm_judge.yaml"
DEFAULT_INPUT_ROOT = ROOT / "data" / "Outputs"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "Outputs_llm"
DATASET_ALIASES = {
    "hitab": "HiTab",
    "mulhi": "MultiHiertt",
    "multihiertt": "MultiHiertt",
}

SYSTEM_PROMPT = """You are a strict answer-equivalence judge for table question answering.
Compare the candidate answer with the reference answer in the context of the question. Do not
independently replace the reference answer. Accept differences that only concern formatting,
wording, ordering where order is irrelevant, harmless rounding, or an explicit unit already
implied by the question. For example, if the question asks for a percentage, 15.5 and 15.5% are
equivalent; if it asks for a dollar amount, 55 and $55 are equivalent. Also accept mathematically
equivalent representations such as 0.55 and 55% when the context supports that conversion.
Reject different values, incompatible units or scales, wrong signs, contradictions, and missing
required items. Return JSON only with this schema:
{"correct": true or false, "confidence": number from 0 to 1, "reason": "brief explanation"}.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluator", default="gpt_oss")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline", action="append", help="Pipeline to evaluate; repeat as needed or use all. Defaults to GraphOtter.")
    parser.add_argument("--dataset", action="append", help="HiTab, MulHi, or MultiHiertt. Defaults to HiTab and MulHi.")
    parser.add_argument("--report", default="latest", help="latest, all, or a report number such as 3.")
    parser.add_argument("--limit", type=int, help="Evaluate at most this many exact-match-wrong results per report.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent judge requests. Defaults to 4.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--skip-errors", action="store_true", help="Keep prior API errors instead of retrying them.")
    parser.add_argument("--overwrite", action="store_true", help="Discard prior judgments for the selected reports.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected reports and one prompt without calling the API.")
    return parser.parse_args()


def resolve_env(value):
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def replace(match: re.Match) -> str:
        return os.environ.get(match.group(1), "")

    return pattern.sub(replace, value)


def load_evaluator(path: Path, name: str) -> tuple[dict, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_config = payload.get(name)
    if not isinstance(raw_config, dict):
        raise ValueError(f'Evaluator "{name}" was not found in {path}.')
    config = resolve_env(raw_config)
    if config.get("provider") != "openai_compatible":
        raise ValueError("Only the openai_compatible provider is supported.")
    required = ("base_url", "model")
    missing = [key for key in required if not str(config.get(key) or "").strip()]
    if missing:
        raise ValueError(f"Missing evaluator configuration: {', '.join(missing)}")
    api_key_template = str(raw_config.get("api_key") or "")
    env_match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", api_key_template)
    return config, env_match.group(1) if env_match else "configured secret"


def normalize_dataset(name: str) -> str:
    normalized = DATASET_ALIASES.get(name.strip().casefold())
    if not normalized:
        raise ValueError(f"Unsupported dataset: {name}")
    return normalized


def report_number(path: Path) -> int:
    suffix = path.stem.removeprefix("report_")
    return int(suffix) if suffix.isdigit() else -1


def select_reports(input_root: Path, pipelines: list[str] | None, datasets: list[str], report: str) -> list[Path]:
    selected_pipelines = pipelines or ["GraphOtter"]
    if any(name.casefold() == "all" for name in selected_pipelines):
        selected_pipelines = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    reports = []
    for pipeline in selected_pipelines:
        for dataset in datasets:
            candidates = sorted((input_root / pipeline / dataset).glob("report_*.json"), key=report_number)
            if report == "latest":
                candidates = candidates[-1:]
            elif report != "all":
                candidates = [path for path in candidates if report_number(path) == int(report)]
            reports.extend(candidates)
    return reports


def answer_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(question, reference, candidate) -> str:
    return "\n".join((
        f"Question: {answer_json(question)}",
        f"Reference answer: {answer_json(reference)}",
        f"Candidate answer: {answer_json(candidate)}",
        "Judge whether the candidate correctly answers the question relative to the reference.",
    ))


def input_hash(raw: dict) -> str:
    content = answer_json([raw.get("question"), raw.get("gold_answer"), raw.get("predicted_answer")])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def exact_match_correct(raw: dict) -> bool:
    try:
        return float(raw.get("exact_match") or 0) >= 1
    except (TypeError, ValueError):
        return False


def needs_llm_judge(raw: dict) -> bool:
    return not raw.get("error") and raw.get("predicted_answer") is not None and not exact_match_correct(raw)


def response_text(payload: dict) -> str:
    content = payload["choices"][0]["message"].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    return str(content)


def parse_correct(value) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "correct", "1"}:
        return True
    if normalized in {"false", "no", "incorrect", "wrong", "0"}:
        return False
    return None


def normalize_judgment(payload: dict) -> dict | None:
    raw_correct = payload.get("correct")
    if raw_correct is None:
        raw_correct = payload.get("verdict")
    correct = parse_correct(raw_correct)
    if correct is None:
        return None
    try:
        confidence = float(payload.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    if 1 < confidence <= 100:
        confidence /= 100
    return {
        "correct": correct,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(payload.get("reason") or payload.get("explanation") or "").strip(),
    }


def parse_judgment(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            normalized = normalize_judgment(payload)
            if normalized:
                return normalized
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            normalized = normalize_judgment(payload)
            if normalized:
                return normalized

    partial_correct = re.search(r'["\']correct["\']\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if partial_correct:
        confidence_match = re.search(r'["\']confidence["\']\s*:\s*([0-9]+(?:\.[0-9]+)?)', cleaned, flags=re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 1.0
        if 1 < confidence <= 100:
            confidence /= 100
        return {
            "correct": partial_correct.group(1).casefold() == "true",
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": "Judge response was truncated after the verdict.",
        }

    plain = parse_correct(cleaned.rstrip(".!"))
    if plain is not None:
        return {"correct": plain, "confidence": 1.0, "reason": ""}
    raise ValueError(f"Judge returned an unsupported response: {cleaned[:300]}")


def chat_completion(config: dict, user_prompt: str) -> tuple[dict, str, float, int]:
    endpoint = f'{str(config["base_url"]).rstrip("/")}/chat/completions'
    request_payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.get("temperature", 0),
    }
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    max_retries = max(0, int(config.get("max_retries", 0)))
    retry_delay = max(0.0, float(config.get("retry_delay_seconds", 1)))
    timeout = max(1.0, float(config.get("timeout_seconds", 100)))
    started = time.monotonic()

    for attempt in range(1, max_retries + 2):
        headers = {"Content-Type": "application/json"}
        if config.get("api_key"):
            headers["Authorization"] = f'Bearer {config["api_key"]}'
        request = Request(endpoint, data=body, method="POST", headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_response = response_text(payload)
            return parse_judgment(raw_response), raw_response, time.monotonic() - started, attempt
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
            if attempt > max_retries:
                raise RuntimeError(f"LLM judge failed after {attempt} attempt(s): {error}") from error
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def result_key(raw: dict, index: int) -> str:
    return str(raw.get("sample_id") or f"index:{index}")


def make_result(raw: dict, index: int, judgment: dict) -> dict:
    result = dict(raw)
    result.update({
        "source_result_index": index,
        "input_hash": input_hash(raw),
        "evaluation_method": "llm_judge",
        "resolved_correct": judgment.get("correct") if isinstance(judgment.get("correct"), bool) else None,
        "llm_judge": judgment,
    })
    return result


def copied_result(raw: dict, index: int) -> dict:
    result = dict(raw)
    if exact_match_correct(raw):
        method = "exact_match"
        resolved_correct = True
    elif raw.get("error") or raw.get("predicted_answer") is None:
        method = "source_error"
        resolved_correct = None
    else:
        method = "pending_llm_judge"
        resolved_correct = None
    result.update({
        "source_result_index": index,
        "input_hash": input_hash(raw),
        "evaluation_method": method,
        "resolved_correct": resolved_correct,
    })
    result.pop("llm_judge", None)
    return result


def mirrored_results(source_results: list[dict], judged_results: dict[str, dict]) -> list[dict]:
    output = []
    for index, raw in enumerate(source_results):
        judged = judged_results.get(result_key(raw, index))
        output.append(judged if judged is not None else copied_result(raw, index))
    return output


def report_summary(results: list[dict], total: int, exact_correct: int = 0, selected: int | None = None, source_errors: int = 0) -> dict:
    judgments = [item.get("llm_judge", {}) for item in results]
    evaluated = [item for item in judgments if isinstance(item.get("correct"), bool)]
    rescued_correct = sum(item["correct"] for item in evaluated)
    skipped = sum(bool(item.get("skipped")) for item in judgments)
    selected = len(results) if selected is None else selected
    combined_correct = exact_correct + rescued_correct
    return {
        "total": total,
        "completed": len(evaluated) + skipped,
        "evaluated": len(evaluated),
        "selected": selected,
        "pending": max(0, selected - len(evaluated) - skipped),
        "exactCorrect": exact_correct,
        "rescoredCorrect": rescued_correct,
        "rescoredWrong": len(evaluated) - rescued_correct,
        "correct": combined_correct,
        "wrong": max(0, total - combined_correct - source_errors),
        "sourceErrors": source_errors,
        "errors": sum(bool(item.get("error")) for item in judgments),
        "skipped": skipped,
        "score": combined_correct / total if total else None,
    }


def write_report(path: Path, payload: dict) -> None:
    summary = report_summary(
        payload["results"],
        payload["source_result_count"],
        payload.get("exact_correct_count", 0),
        payload.get("selected_result_count"),
        payload.get("source_error_count", 0),
    )
    payload.update(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def existing_results(path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        result_key(item, int(item.get("source_result_index", index))): item
        for index, item in enumerate(payload.get("results", []))
        if isinstance(item, dict)
    }


def evaluate_report(source_path: Path, output_path: Path, config: dict, api_key_env: str, args: argparse.Namespace) -> None:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    all_source_results = source.get("results", [])
    selected_results = [(index, raw) for index, raw in enumerate(all_source_results) if needs_llm_judge(raw)]
    selected_result_count = len(selected_results)
    if args.limit is not None:
        selected_results = selected_results[:max(0, args.limit)]
    prior = existing_results(output_path, args.overwrite)
    results_by_key: dict[str, dict] = {}
    relative_source = source_path.relative_to(ROOT).as_posix()
    payload = {
        "version": 1,
        "evaluator": {
            "name": args.evaluator,
            "provider": config["provider"],
            "base_url": config["base_url"],
            "model": config["model"],
            "temperature": config.get("temperature", 0),
            "timeout_seconds": config.get("timeout_seconds", 100),
            "max_retries": config.get("max_retries", 2),
            "retry_delay_seconds": config.get("retry_delay_seconds", 1),
            "api_key_env": api_key_env,
        },
        "source_report": relative_source,
        "selection": "exact_match_wrong_only",
        "source_result_count": len(all_source_results),
        "selected_result_count": selected_result_count,
        "scheduled_result_count": len(selected_results),
        "exact_correct_count": sum(exact_match_correct(raw) for raw in all_source_results),
        "source_error_count": sum(bool(raw.get("error") or raw.get("predicted_answer") is None) for raw in all_source_results),
        "pipeline": source.get("pipeline") or source_path.parent.parent.name,
        "dataset": source.get("dataset") or source_path.parent.name,
        "run_id": source.get("run_id"),
        "run_name": source.get("run_name"),
        "results": [],
    }

    pending_results = []
    for index, raw in selected_results:
        key = result_key(raw, index)
        cached = prior.get(key)
        cached_judge = cached.get("llm_judge", {}) if cached else {}
        cached_complete = isinstance(cached_judge.get("correct"), bool) or cached_judge.get("skipped")
        if args.skip_errors and cached_judge.get("error"):
            cached_complete = True
        if cached and cached.get("input_hash") == input_hash(raw) and cached_complete:
            results_by_key[key] = make_result(raw, index, cached_judge)
            continue

        pending_results.append((index, raw))

    def judge_result(item: tuple[int, dict]) -> tuple[int, dict, dict]:
        index, raw = item

        try:
            judgment, raw_response, latency, attempts = chat_completion(
                config,
                build_user_prompt(raw.get("question"), raw.get("gold_answer"), raw.get("predicted_answer")),
            )
            judgment.update({
                "raw_response": raw_response,
                "latency_seconds": round(latency, 3),
                "attempts": attempts,
            })
        except RuntimeError as error:
            judgment = {
                "correct": None,
                "confidence": 0.0,
                "reason": "The evaluator request failed.",
                "error": str(error),
            }
        return index, raw, judgment

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        judged_results = executor.map(judge_result, pending_results)
        for processed, (index, raw, judgment) in enumerate(judged_results, 1):
            key = result_key(raw, index)
            results_by_key[key] = make_result(raw, index, judgment)
            payload["results"] = mirrored_results(all_source_results, results_by_key)
            verdict = judgment.get("correct")
            print(f"  [{len(results_by_key)}/{len(selected_results)}] {key}: {verdict if verdict is not None else 'unscored'}", flush=True)
            if processed % max(1, args.checkpoint_every) == 0:
                write_report(output_path, payload)

    payload["results"] = mirrored_results(all_source_results, results_by_key)
    write_report(output_path, payload)
    summary = report_summary(
        payload["results"],
        len(all_source_results),
        payload["exact_correct_count"],
        selected_result_count,
        payload["source_error_count"],
    )
    score = "-" if summary["score"] is None else f'{summary["score"]:.4f}'
    print(f"  Wrote {output_path.relative_to(ROOT)}: score={score}, rescored={summary['evaluated']}/{selected_result_count}, errors={summary['errors']}")


def main() -> int:
    args = parse_args()
    try:
        args.config = args.config.resolve()
        args.input_root = args.input_root.resolve()
        args.output_root = args.output_root.resolve()
        datasets = [normalize_dataset(item) for item in (args.dataset or ["HiTab", "MulHi"])]
        reports = select_reports(args.input_root, args.pipeline, datasets, args.report)
        if not reports:
            raise ValueError("No matching report files were found.")
        if args.dry_run:
            print("Selected reports:")
            for path in reports:
                print(f"  {path.relative_to(ROOT)}")
            source_results = json.loads(reports[0].read_text(encoding="utf-8")).get("results", [])
            wrong_results = [raw for raw in source_results if needs_llm_judge(raw)]
            if not wrong_results:
                raise ValueError("The selected report has no exact-match-wrong answers to judge.")
            sample = wrong_results[0]
            print(f"Exact-match-wrong cases in first report: {len(wrong_results)}")
            print("\nSystem prompt:\n" + SYSTEM_PROMPT)
            print("User prompt:\n" + build_user_prompt(sample.get("question"), sample.get("gold_answer"), sample.get("predicted_answer")))
            return 0

        config, api_key_env = load_evaluator(args.config, args.evaluator)
        for source_path in reports:
            relative = source_path.relative_to(args.input_root)
            output_path = args.output_root / relative
            print(f"Evaluating {relative} -> {output_path.relative_to(ROOT)}", flush=True)
            evaluate_report(source_path, output_path, config, api_key_env, args)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
