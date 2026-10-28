# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the length-bucketing ladder (pure CPU, no model import)."""

import pytest

from vllm_omni.model_executor.bucketed_encoder.bucketing import (
    LengthBucketingConfig,
    parse_seconds_ladder,
)


def test_padded_length_rounds_up_to_smallest_rung():
    cfg = LengthBucketingConfig(buckets=(500, 1000, 2000, 4000), max_length=4000)
    assert cfg.padded_length(0) == 500
    assert cfg.padded_length(499) == 500
    assert cfg.padded_length(500) == 500
    assert cfg.padded_length(501) == 1000
    assert cfg.padded_length(2000) == 2000
    assert cfg.padded_length(3999) == 4000
    assert cfg.padded_length(4000) == 4000
    # Past the ladder: capped at max_length, never above it.
    assert cfg.padded_length(99999) == 4000


def test_padded_length_caps_at_max_length_even_if_bucket_exceeds_it():
    # A shared global ladder may carry rungs above this model's cap; they must
    # never be selected.
    cfg = LengthBucketingConfig(buckets=(1000, 2000, 4000), max_length=2000)
    assert cfg.padded_length(1500) == 2000
    assert cfg.padded_length(2500) == 2000


def test_rejects_unsorted_or_empty_buckets():
    with pytest.raises(ValueError):
        LengthBucketingConfig(buckets=(2000, 1000), max_length=4000)
    with pytest.raises(ValueError):
        LengthBucketingConfig(buckets=(), max_length=4000)
    with pytest.raises(ValueError):
        LengthBucketingConfig(buckets=(1000,), max_length=0)
    with pytest.raises(ValueError):
        LengthBucketingConfig(buckets=(1000,), max_length=4000, min_length=-1)


def test_feature_lengths_drops_unreachable_rungs():
    cfg = LengthBucketingConfig(buckets=(500, 1000, 2000, 4000), max_length=2000)

    def to_feature_length(samples: int) -> int:
        return samples // 500

    assert cfg.feature_lengths(to_feature_length) == (1, 2, 4)


def test_parse_seconds_ladder_default_and_rounding():
    cfg = parse_seconds_ladder(
        "",
        sample_rate=16000,
        default_seconds=(0.5, 1, 2, 4, 8, 16, 30),
        max_seconds=30.0,
    )
    assert cfg.buckets == (8000, 16000, 32000, 64000, 128000, 256000, 480000)
    assert cfg.max_length == 480000
    # 1 s audio -> 1 s bucket, not 30 s.
    assert cfg.padded_length(16000) == 16000
    assert cfg.padded_length(16001) == 32000


def test_parse_seconds_ladder_env_style():
    cfg = parse_seconds_ladder(
        "0.5,2,4,30",
        sample_rate=16000,
        default_seconds=(),
        max_seconds=30.0,
    )
    assert cfg.buckets == (8000, 32000, 64000, 480000)


def test_parse_seconds_ladder_validation():
    with pytest.raises(ValueError):
        parse_seconds_ladder("", sample_rate=16000, default_seconds=(),
                             max_seconds=30.0)  # empty ladder
    with pytest.raises(ValueError):
        parse_seconds_ladder("0.01", sample_rate=16000, default_seconds=(),
                             min_seconds=0.3, max_seconds=30.0)  # too small
    with pytest.raises(ValueError):
        parse_seconds_ladder("0.5,60", sample_rate=16000, default_seconds=(),
                             max_seconds=30.0)  # above cap
    with pytest.raises(ValueError):
        parse_seconds_ladder("0.5,abc", sample_rate=16000, default_seconds=(),
                             max_seconds=30.0)  # unparseable
