from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

_TYPICAL_MAX_TOKENS: float = 300.0
_MAX_CHAR_ENTROPY: float = math.log2(95)
_MAX_TOKEN_ENTROPY: float = math.log2(300)

_ALPHA: float = 0.40
_BETA:  float = 0.30
_GAMMA: float = 0.30

@dataclass
class MDLMetrics:
    token_count:       int
    char_entropy:      float
    token_entropy:     float
    zlib_ratio:        float
    zlib_bytes:        int
    raw_text_bytes:    int
    lm_score:          float
    ld_given_m_score:  float
    total_mdl:         float
    asr:               float = 0.0
    frr:               float = 0.0

def tokenise(text: str) -> list[str]:
    return text.split()

def shannon_entropy_chars(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )

def shannon_entropy_tokens(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )

def zlib_compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)

class MDLCalculator:

    def compute(
        self,
        text: str,
        asr: float = 0.0,
        frr: float = 0.0,
    ) -> MDLMetrics:
        tokens = tokenise(text)
        token_count = len(tokens)

        char_entropy  = shannon_entropy_chars(text)
        token_entropy = shannon_entropy_tokens(tokens)

        raw_bytes  = len(text.encode("utf-8"))
        z_ratio    = zlib_compression_ratio(text)
        z_bytes    = len(zlib.compress(text.encode("utf-8"), level=9)) if raw_bytes else 0

        norm_token_count   = min(token_count   / _TYPICAL_MAX_TOKENS, 1.0)
        norm_char_entropy  = min(char_entropy  / _MAX_CHAR_ENTROPY,   1.0)

        norm_zlib_penalty  = 1.0 - min(z_ratio, 1.0)

        lm_score = (
            _ALPHA * norm_token_count
            + _BETA  * norm_char_entropy
            + _GAMMA * norm_zlib_penalty
        )
        lm_score = float(np.clip(lm_score, 0.0, 1.0))

        ld_given_m = float(np.clip((asr + frr) / 2.0, 0.0, 1.0))

        total_mdl = float(np.clip(0.6 * lm_score + 0.4 * ld_given_m, 0.0, 1.0))

        return MDLMetrics(
            token_count=token_count,
            char_entropy=round(char_entropy,  4),
            token_entropy=round(token_entropy, 4),
            zlib_ratio=round(z_ratio,         4),
            zlib_bytes=z_bytes,
            raw_text_bytes=raw_bytes,
            lm_score=round(lm_score,          4),
            ld_given_m_score=round(ld_given_m, 4),
            total_mdl=round(total_mdl,         4),
            asr=asr,
            frr=frr,
        )

    def compression_index(self, text: str) -> float:
        return zlib_compression_ratio(text)

    def batch_compute(
        self,
        texts: list[str],
        asrs: list[float],
        frrs: list[float],
    ) -> list[MDLMetrics]:
        assert len(texts) == len(asrs) == len(frrs), (
            "texts, asrs, and frrs must be the same length."
        )
        return [
            self.compute(t, a, f)
            for t, a, f in zip(texts, asrs, frrs)
        ]
