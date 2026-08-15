# tests/test_verify_ablation_isolation.py
from scripts.verify_ablation_isolation import verify_arm_isolation


class TestVerifyArmIsolation:
    def test_arm1_valid(self):
        samples = [
            {
                "sample_id": "s1",
                "telemetry": {
                    "multi_query_count": 1,
                    "hyde_generated": False,
                    "stepback_generated": False,
                    "reranked": False,
                    "chunk_sources": ["dense"],
                    "graph_chunks_count": 0,
                    "crag_action": None,
                },
            }
        ]
        violations = verify_arm_isolation(1, "arm_1_base", samples)
        assert violations == []

    def test_arm1_detects_reranker_leakage(self):
        samples = [
            {
                "sample_id": "s1",
                "telemetry": {
                    "multi_query_count": 1,
                    "hyde_generated": False,
                    "stepback_generated": False,
                    "reranked": True,  # LEAKAGE
                    "chunk_sources": ["dense"],
                    "graph_chunks_count": 0,
                    "crag_action": None,
                },
            }
        ]
        violations = verify_arm_isolation(1, "arm_1_base", samples)
        assert len(violations) == 1
        assert "Reranker leakage" in violations[0]

    def test_arm1_detects_bm25_leakage(self):
        samples = [
            {
                "sample_id": "s1",
                "telemetry": {
                    "multi_query_count": 1,
                    "hyde_generated": False,
                    "stepback_generated": False,
                    "reranked": False,
                    "chunk_sources": ["dense", "bm25"],  # LEAKAGE
                    "graph_chunks_count": 0,
                    "crag_action": None,
                },
            }
        ]
        violations = verify_arm_isolation(1, "arm_1_base", samples)
        assert len(violations) == 1
        assert "Sparse/Graph chunk leakage" in violations[0]

    def test_arm2_detects_transform_leakage(self):
        samples = [
            {
                "sample_id": "s2",
                "telemetry": {
                    "multi_query_count": 4,  # LEAKAGE
                    "hyde_generated": True,  # LEAKAGE
                    "stepback_generated": False,
                    "reranked": False,
                    "chunk_sources": ["dense", "bm25"],
                    "graph_chunks_count": 0,
                    "crag_action": None,
                },
            }
        ]
        violations = verify_arm_isolation(2, "arm_2_bm25", samples)
        assert len(violations) == 2
        assert any("Multi-query leakage" in v for v in violations)
        assert any("HyDE leakage" in v for v in violations)

    def test_arm4_detects_missing_reranker(self):
        samples = [
            {
                "sample_id": "s4",
                "telemetry": {
                    "multi_query_count": 4,
                    "hyde_generated": True,
                    "stepback_generated": True,
                    "reranked": False,  # MISSING
                    "chunk_sources": ["dense", "bm25"],
                    "graph_chunks_count": 0,
                    "crag_action": None,
                },
            }
        ]
        violations = verify_arm_isolation(4, "arm_4_flashrank", samples)
        assert len(violations) == 1
        assert "Reranker was NOT executed" in violations[0]

    def test_arm6_valid(self):
        samples = [
            {
                "sample_id": "s6",
                "telemetry": {
                    "multi_query_count": 4,
                    "hyde_generated": True,
                    "stepback_generated": True,
                    "reranked": True,
                    "chunk_sources": ["dense", "bm25", "graph"],
                    "graph_chunks_count": 1,
                    "crag_action": "correct",
                },
            }
        ]
        violations = verify_arm_isolation(6, "arm_6_crag", samples)
        assert violations == []
