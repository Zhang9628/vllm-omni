# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reusable length-bucketing + CUDA-graph pooling for variable-length encoders.

This package is encoder-agnostic: it only assumes the encoder takes a single
forward pass over an independent batch of variable-length inputs (no
BatchNorm / cross-attention between samples). It provides

* :class:`LengthBucketingConfig` (``bucketing.py``) — a discrete length
  ladder with a ``padded_length`` lookup, used to pad short inputs up to the
  smallest rung instead of a single global maximum;
* :class:`CUDAGraphRunner` (``cuda_graph_pool.py``) — a thin wrapper that
  captures/replays one CUDA graph for one fixed input shape;
* :class:`BucketedCUDAGraphRunner` (``cuda_graph_pool.py``) — the pool
  manager that captures one graph per ``(batch_size, length_rung)`` pair into
  a single shared memory pool, then dispatches by lookup with a chunked eager
  fallback.
"""

from vllm_omni.model_executor.bucketed_encoder.bucketing import (
    LengthBucketingConfig,
    parse_seconds_ladder,
)
from vllm_omni.model_executor.bucketed_encoder.cuda_graph_pool import (
    BucketedCUDAGraphRunner,
    CUDAGraphRunner,
)

__all__ = [
    "BucketedCUDAGraphRunner",
    "CUDAGraphRunner",
    "LengthBucketingConfig",
    "parse_seconds_ladder",
]
