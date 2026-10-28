# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Length bucketing for variable-length encoder inputs.

Encoders whose cost scales with the *padded* input length (audio feature
extractors, vision patch encoders, ...) are commonly padded to a single
global maximum length. A 1 s utterance inside a 30 s audio model then costs
the same as a 30 s utterance: up to ``max_length / min_length`` x wasted
compute and H2D traffic.

The fix is a discrete "length ladder": pad each sample up to the *smallest*
rung ``>=`` its real length instead of the global max. This module exposes

* the ladder itself (:class:`LengthBucketingConfig`), in whatever unit the
  caller uses for encoder input lengths (samples, feature frames, patches,
  tokens ...); and
* a small helper (:func:`parse_seconds_ladder`) for the common case where the
  ladder is expressed in seconds at a fixed sampling rate.

The ladder rungs are intentionally shared with
:class:`~vllm_omni.model_executor.bucketed_encoder.cuda_graph_pool.BucketedCUDAGraphRunner`:
CUDA graphs need static shapes, and a bucketed ladder guarantees the encoder
input length only ever takes the rung values, so the graph pool only has to
capture one graph per rung instead of one per possible length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LengthBucketingConfig:
    """A discrete length ladder used to bucket variable-length inputs.

    The ladder is expressed in the caller's native "length" unit. For an
    audio frontend that usually means waveform samples or mel feature frames;
    for a vision encoder it could be pixel patches along one axis. The only
    requirement is that ``padded_length`` is called with a value in the same
    unit as ``buckets``.

    Args:
        buckets: Ascending ladder rungs. Every padded length collapses to one
            of these values (or ``max_length``).
        max_length: Hard upper bound on the padded length. Padding never
            exceeds this, so ``buckets`` may contain rungs above it and they
            will simply never be selected (safe to share a global ladder).
        min_length: Lower bound on a meaningful input length. Purely
            informational; ``padded_length`` still accepts shorter values.
    """

    buckets: tuple[int, ...]
    max_length: int
    min_length: int = 0

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("LengthBucketingConfig.buckets must be non-empty")
        if tuple(self.buckets) != tuple(sorted(self.buckets)):
            raise ValueError(
                "LengthBucketingConfig.buckets must be sorted ascending, "
                f"got {self.buckets}")
        if self.min_length < 0:
            raise ValueError(
                f"LengthBucketingConfig.min_length must be >= 0, got "
                f"{self.min_length}")
        if self.max_length <= 0:
            raise ValueError(
                f"LengthBucketingConfig.max_length must be > 0, got "
                f"{self.max_length}")

    def padded_length(self, length: int) -> int:
        """Return the smallest rung ``>=`` ``length``, capped at ``max_length``.

        Args:
            length: The real (unpadded) input length, in the same unit as
                ``buckets``.

        Returns:
            The length the input should be padded to.
        """
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        for rung in self.buckets:
            if rung >= length:
                return min(rung, self.max_length)
        return self.max_length

    def feature_lengths(self, to_feature_length) -> tuple[int, ...]:
        """Map every *reachable* rung to a feature length via ``to_feature_length``.

        ``to_feature_length`` is a model-specific callable (e.g. the feature
        extractor's ``samples -> frames`` formula). Rungs above ``max_length``
        are dropped, since ``padded_length`` never selects them. The result is
        the ladder a ``BucketedCUDAGraphRunner`` should capture graphs for.
        """
        reachable = [r for r in self.buckets if r <= self.max_length]
        return tuple(sorted({to_feature_length(r) for r in reachable}))


def parse_seconds_ladder(
    raw: str,
    *,
    sample_rate: int,
    default_seconds: Sequence[float],
    min_seconds: float = 0.0,
    max_seconds: float,
) -> LengthBucketingConfig:
    """Parse a comma-separated seconds ladder into a sample-rung config.

    This is the canonical way to make the ladder deploy-time tunable via an
    environment variable without touching code: ``raw`` is typically read
    from ``os.environ`` and, when empty, falls back to ``default_seconds``.

    Args:
        raw: Comma-separated seconds (e.g. ``"0.5,1,2,4,8,16,30"``). Empty
            string selects ``default_seconds``.
        sample_rate: Samples per second; each second is rounded to an integer
            sample count.
        default_seconds: Ladder used when ``raw`` is empty.
        min_seconds: Minimum allowed rung; a smaller rung raises.
        max_seconds: Maximum allowed rung and the ``max_length`` cap. Rungs
            above this are rejected (they can never be hit).

    Returns:
        A :class:`LengthBucketingConfig` whose ``buckets`` are in samples.

    Raises:
        ValueError: If ``raw`` cannot be parsed, is empty, or contains a rung
            outside ``[min_seconds, max_seconds]``.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    seconds: Sequence[float]
    if not raw.strip():
        seconds = tuple(default_seconds)
    else:
        try:
            seconds = tuple(sorted({float(x) for x in raw.split(",") if x.strip()}))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse seconds ladder {raw!r}: expected "
                f"comma-separated numbers (e.g. 0.5,1,2,4,8,16,30)") from exc

    if not seconds:
        raise ValueError("seconds ladder must not be empty")
    if seconds[0] < min_seconds:
        raise ValueError(
            f"Smallest rung {seconds[0]}s is below min_seconds={min_seconds}s")
    if seconds[-1] > max_seconds:
        raise ValueError(
            f"Largest rung {seconds[-1]}s exceeds max_seconds={max_seconds}s")

    buckets = tuple(int(round(s * sample_rate)) for s in seconds)
    # Guard against duplicate rungs after rounding (e.g. 0.001s @ 1000 Hz).
    buckets = tuple(sorted(set(buckets)))
    if len(buckets) < 2:
        raise ValueError(
            f"seconds ladder {seconds!r} collapses to fewer than 2 sample "
            f"rungs after rounding at sample_rate={sample_rate}")

    return LengthBucketingConfig(
        buckets=buckets,
        max_length=int(round(max_seconds * sample_rate)),
        min_length=int(round(min_seconds * sample_rate)),
    )
