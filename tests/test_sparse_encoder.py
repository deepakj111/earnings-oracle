from ingestion.sparse_encoder import SparseEncoder


def test_sparse_encoder_deterministic():
    text = "NVIDIA reported revenue of $35.1 billion for the fourth quarter."
    vec1 = SparseEncoder.encode(text)
    vec2 = SparseEncoder.encode(text)
    assert vec1.indices == vec2.indices
    assert vec1.values == vec2.values
    assert len(vec1.indices) > 0
    assert len(vec1.values) == len(vec1.indices)


def test_sparse_encoder_empty():
    vec = SparseEncoder.encode("")
    assert vec.indices == []
    assert vec.values == []


def test_sparse_encoder_to_qdrant():
    q_vec = SparseEncoder.encode_query("Data Center gross margin")
    assert hasattr(q_vec, "indices")
    assert hasattr(q_vec, "values")
    assert len(q_vec.indices) > 0
