import json
import tempfile
import unittest
from pathlib import Path

from mycelium.pipeline.patch_batch_logger import PatchBatchLogger


class TestPatchBatchLogger(unittest.TestCase):
    def test_logs_to_domain_partition_and_fills_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PatchBatchLogger(batch_dir=Path(tmpdir))
            trace_id = "trace-123"

            logger.log_query(
                trace_id=trace_id,
                query="Explain string theory",
                tags=["string", "theory"],
                layer0_route="REASONING_PIPELINE",
                routing_classification="ATTRIBUTE_ONLY",
                domains=[],
                expert_decision="CREATE_NEW_PATCH",
                confidence=0.0,
                phase_latencies_ms={"phase_3_validation": 100.0},
                metadata={"domain_tag": "unclassified"},
            )

            files = list(Path(tmpdir).rglob("*.jsonl"))
            self.assertEqual(1, len(files))
            self.assertIn("unclassified", str(files[0]))

            logger.fill_response(
                trace_id,
                "String theory is a framework...",
                sandbox_evidence=[{"tool": "paper_search", "query": "string theory"}],
                phase_latencies_ms={"phase_3_validation": 101.0},
            )

            record = json.loads(files[0].read_text(encoding="utf-8").strip())
            self.assertEqual(trace_id, record["trace_id"])
            self.assertEqual("String theory is a framework...", record["response"])
            self.assertEqual("ATTRIBUTE_ONLY", record["routing"]["classification"])
            self.assertEqual("CREATE_NEW_PATCH", record["routing"]["expert_decision"])
            self.assertEqual(0.0, record["routing"]["confidence"])
            self.assertEqual("paper_search", record["sandbox_evidence"][0]["tool"])
            self.assertEqual(101.0, record["phase_latencies_ms"]["phase_3_validation"])


if __name__ == "__main__":
    unittest.main()
