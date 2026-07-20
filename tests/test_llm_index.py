import json
import tempfile
import unittest
from pathlib import Path

from api import server


class LlmIndexTests(unittest.TestCase):
    def setUp(self):
        self.original_data = server.DATA
        self.original_llm_root = server.OUTPUTS_LLM_ROOT
        self.temporary = tempfile.TemporaryDirectory()
        server.DATA = Path(self.temporary.name)
        server.OUTPUTS_LLM_ROOT = server.DATA / "Outputs_llm"
        server.artifact_catalog.cache_clear()

    def tearDown(self):
        server.DATA = self.original_data
        server.OUTPUTS_LLM_ROOT = self.original_llm_root
        server.artifact_catalog.cache_clear()
        self.temporary.cleanup()

    def test_llm_verdict_overrides_exact_match_and_adds_report_artifact(self):
        source_path = server.DATA / "Outputs" / "GraphOtter" / "HiTab" / "report_3.json"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(json.dumps({
            "pipeline": "GraphOtter",
            "dataset": "HiTab",
            "results": [{
                "sample_id": "sample-1",
                "question": "What percent?",
                "gold_answer": [15.5],
                "predicted_answer": "15.5%",
                "exact_match": 0,
                "error": False,
            }],
        }), encoding="utf-8")

        judge_path = server.OUTPUTS_LLM_ROOT / "GraphOtter" / "HiTab" / "report_3.json"
        judge_path.parent.mkdir(parents=True)
        judge_path.write_text(json.dumps({
            "source_report": "data/Outputs/GraphOtter/HiTab/report_3.json",
            "score": 1.0,
            "evaluated": 1,
            "source_result_count": 1,
            "evaluator": {"name": "gpt_oss", "model": "gpt-oss-20b"},
            "results": [{
                "sample_id": "sample-1",
                "source_result_index": 0,
                "evaluation_method": "llm_judge",
                "llm_judge": {"correct": True, "confidence": 0.98, "reason": "The question supplies the percent unit."},
            }],
        }), encoding="utf-8")

        records = server.load_output_records("GraphOtter", "HiTab")

        self.assertEqual(records[0]["status"], "correct")
        self.assertTrue(records[0]["evaluation"]["disagreesWithExactMatch"])
        self.assertEqual(records[0]["evaluation"]["reportScore"], 1.0)
        self.assertIn("Outputs_llm/GraphOtter/HiTab/report_3.json", records[0]["artifacts"]["output"])

    def test_copied_exact_match_row_uses_source_status_without_llm_badge(self):
        source_path = server.DATA / "Outputs" / "GraphOtter" / "HiTab" / "report_3.json"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(json.dumps({
            "results": [{
                "sample_id": "sample-2",
                "question": "What percent?",
                "gold_answer": [52.1],
                "predicted_answer": "52.1",
                "exact_match": 1,
                "error": False,
            }],
        }), encoding="utf-8")
        judge_path = server.OUTPUTS_LLM_ROOT / "GraphOtter" / "HiTab" / "report_3.json"
        judge_path.parent.mkdir(parents=True)
        judge_path.write_text(json.dumps({
            "source_report": "data/Outputs/GraphOtter/HiTab/report_3.json",
            "evaluator": {"name": "gpt_oss", "model": "gpt-oss-20b"},
            "results": [{
                "sample_id": "sample-2",
                "source_result_index": 0,
                "evaluation_method": "exact_match",
                "resolved_correct": True,
            }],
        }), encoding="utf-8")

        record = server.load_output_records("GraphOtter", "HiTab")[0]

        self.assertEqual(record["status"], "correct")
        self.assertIsNone(record["evaluation"])
        self.assertIn("Outputs_llm/GraphOtter/HiTab/report_3.json", record["artifacts"]["output"])


if __name__ == "__main__":
    unittest.main()
