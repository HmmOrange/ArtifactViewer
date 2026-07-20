import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_outputs_llm.py"
SPEC = importlib.util.spec_from_file_location("evaluate_outputs_llm", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class LlmEvaluatorTests(unittest.TestCase):
    def test_parse_json_judgment(self):
        result = evaluator.parse_judgment('{"correct": true, "confidence": 0.82, "reason": "Same percent."}')
        self.assertTrue(result["correct"])
        self.assertEqual(result["confidence"], 0.82)

    def test_parse_fenced_judgment_and_percent_confidence(self):
        result = evaluator.parse_judgment('```json\n{"verdict":"incorrect","confidence":75,"explanation":"Different value"}\n```')
        self.assertFalse(result["correct"])
        self.assertEqual(result["confidence"], 0.75)

    def test_parse_truncated_json_after_verdict(self):
        result = evaluator.parse_judgment('{"correct": true, "confidence": 0.99, "reason": "Same')
        self.assertTrue(result["correct"])
        self.assertEqual(result["confidence"], 0.99)

    def test_prompt_includes_question_and_both_answers(self):
        prompt = evaluator.build_user_prompt("What percent?", [55.0], "55%")
        self.assertIn('Question: "What percent?"', prompt)
        self.assertIn("Reference answer: [55.0]", prompt)
        self.assertIn('Candidate answer: "55%"', prompt)

    def test_evaluator_allows_an_empty_api_key(self):
        original = os.environ.pop("GPT_OSS_API_KEY", None)
        try:
            config, api_key_env = evaluator.load_evaluator(evaluator.DEFAULT_CONFIG, "gpt_oss")
        finally:
            if original is not None:
                os.environ["GPT_OSS_API_KEY"] = original
        self.assertEqual(config["api_key"], "")
        self.assertEqual(api_key_env, "GPT_OSS_API_KEY")

    def test_report_summary_ignores_unscored_errors(self):
        results = [
            {"llm_judge": {"correct": True}},
            {"llm_judge": {"correct": False}},
            {"llm_judge": {"correct": None, "error": "timeout"}},
        ]
        summary = evaluator.report_summary(results, 5, exact_correct=2, selected=3)
        self.assertEqual(summary["evaluated"], 2)
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["score"], 0.6)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["errors"], 1)

    def test_evaluate_report_writes_resumable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Outputs" / "GraphOtter" / "HiTab" / "report_3.json"
            output = root / "Outputs_llm" / "GraphOtter" / "HiTab" / "report_3.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({
                "pipeline": "GraphOtter",
                "dataset": "HiTab",
                "results": [{
                    "sample_id": "exact-correct",
                    "question": "What percent?",
                    "gold_answer": [52.1],
                    "predicted_answer": "52.1",
                    "exact_match": 1,
                }, {
                    "sample_id": "needs-judge",
                    "question": "What percent?",
                    "gold_answer": [55.0],
                    "predicted_answer": "55%",
                    "exact_match": 0,
                }],
            }), encoding="utf-8")
            original_root = evaluator.ROOT
            evaluator.ROOT = root
            original_chat = evaluator.chat_completion
            evaluator.chat_completion = lambda config, prompt: (
                {"correct": True, "confidence": 0.9, "reason": "The percent unit is implied."},
                '{"correct":true,"confidence":0.9}',
                0.01,
                1,
            )
            try:
                args = SimpleNamespace(
                    evaluator="gpt_oss",
                    overwrite=False,
                    skip_errors=False,
                    limit=None,
                    checkpoint_every=1,
                    workers=2,
                )
                evaluator.evaluate_report(source, output, {
                    "provider": "openai_compatible",
                    "base_url": "http://example.test/v1",
                    "model": "gpt-oss-20b",
                    "api_key": "key",
                    "temperature": 0,
                    "timeout_seconds": 100,
                    "max_retries": 2,
                    "retry_delay_seconds": 1,
                }, "GPT_OSS_API_KEY", args)
            finally:
                evaluator.chat_completion = original_chat
                evaluator.ROOT = original_root
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["score"], 1.0)
            self.assertEqual(payload["selection"], "exact_match_wrong_only")
            self.assertEqual(payload["exactCorrect"], 1)
            self.assertEqual(payload["selected"], 1)
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["results"][0]["evaluation_method"], "exact_match")
            self.assertEqual(payload["results"][1]["sample_id"], "needs-judge")
            self.assertTrue(payload["results"][1]["llm_judge"]["correct"])


if __name__ == "__main__":
    unittest.main()
