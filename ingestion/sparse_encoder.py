"""
Deterministic Sparse Vector Encoder for Qdrant Native Sparse Hybrid Search.

Converts arbitrary text into Qdrant SparseVector representation (indices, values)
using deterministic token hashing and sublinear term frequency weighting.
Eliminates local in-memory BM25 pickle files (bm25_index.pkl, bm25_corpus.pkl),
enabling fully stateless, horizontally scalable multi-worker deployments.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass

from qdrant_client.http import models as qmodels

from ingestion.indexer import _tokenize_for_bm25


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]

    def to_qdrant(self) -> qmodels.SparseVector:
        return qmodels.SparseVector(indices=self.indices, values=self.values)


class SparseEncoder:
    """
    Encodes text into sparse vector format compatible with Qdrant native sparse search.
    Uses 31-bit CRC32 token hashing for zero-dependency, deterministic vocabulary projection.
    """

    _MAX_INDEX = 0x7FFFFFFF  # 31-bit positive int

    @classmethod
    def encode(cls, text: str) -> SparseVectorData:
        """
        Encode text into a SparseVectorData with sorted unique indices and TF weights.
        """
        tokens = _tokenize_for_bm25(text)
        if not tokens:
            return SparseVectorData(indices=[], values=[])

        tf = Counter(tokens)
        index_to_val: dict[int, float] = {}

        for token, count in tf.items():
            # 31-bit CRC32 hash provides uniform, deterministic hash distribution
            idx = zlib.crc32(token.encode("utf-8")) & cls._MAX_INDEX
            # Sublinear term frequency weighting: 1 + ln(tf)
            val = round(1.0 + math.log(count), 4)
            index_to_val[idx] = max(index_to_val.get(idx, 0.0), val)

        sorted_pairs = sorted(index_to_val.items(), key=lambda p: p[0])
        indices = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        return SparseVectorData(indices=indices, values=values)

    @classmethod
    def encode_query(cls, query: str) -> qmodels.SparseVector:
        """
        Encode query string into Qdrant SparseVector model.
        """
        data = cls.encode(query)
        return data.to_qdrant()
