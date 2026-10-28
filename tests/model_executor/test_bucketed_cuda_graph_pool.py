# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical-equivalence tests for the bucketed CUDA-graph pool.

A tiny deterministic encoder (two linear layers + relu + mask multiply, no
BatchNorm and no randomness) exercises every dispatch path of
:class:`BucketedCUDAGraphRunner`:

* graph replay for an exact ``(batch, length)`` bucket,
* batch padding to the nearest captured batch size,
* large-batch capture on short lengths,
* chunked eager fallback for an unseen length rung,
* oversize-batch chunking.

Each path is compared against a plain eager forward on the same inputs. The
encoder is independent across samples within a batch, so graph replay must
match eager up to floating-point tolerance.
"""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.bucketed_encoder.cuda_graph_pool import (
    BucketedCUDAGraphRunner,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}),
]

DEVICE = torch.device("cuda:0")
FEAT_DIM = 16
HIDDEN_DIM = 16


class DummyEncoder(nn.Module):
    """Deterministic single-forward encoder; independent across batch rows."""

    def __init__(self, feat_dim: int = FEAT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj1 = nn.Linear(feat_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, feat_dim)

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # features: (B, T, feat); mask: (B, T)
        x = torch.relu(self.proj1(features))
        x = self.proj2(x)
        return x * mask.unsqueeze(-1)


@pytest.fixture(scope="module")
def runner() -> BucketedCUDAGraphRunner:
    torch.manual_seed(0)
    model = DummyEncoder().to(DEVICE)

    def make_inputs(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.randn(batch, length, FEAT_DIM, device=DEVICE)
        mask = torch.rand(batch, length, device=DEVICE)
        return features, mask

    return BucketedCUDAGraphRunner(
        model,
        length_buckets=(8, 16, 32),
        make_inputs=make_inputs,
        capture_sizes=(1, 2, 4),
        large_capture_sizes=(8,),
        large_length_max=16,
        eager_chunk=8,
    )


def _run(runner: BucketedCUDAGraphRunner, batch: int, length: int):
    torch.manual_seed(1234)
    features = torch.randn(batch, length, FEAT_DIM, device=DEVICE)
    mask = torch.rand(batch, length, device=DEVICE)
    with torch.inference_mode():
        eager = runner.model(features, mask)
        (graph,) = runner.execute(features, mask)
    return features, mask, eager, graph


def test_shared_memory_pool(runner: BucketedCUDAGraphRunner):
    # Every (batch, length) graph must share one memory pool so peak activation
    # memory stays at a single largest capture instead of the sum.
    pools = {id(r.graph.pool()) for r in runner.graph_runners.values()}
    assert len(pools) == 1


def test_graph_replay_exact_bucket(runner: BucketedCUDAGraphRunner):
    _, _, eager, graph = _run(runner, batch=2, length=8)
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)


def test_batch_padding_to_captured_size(runner: BucketedCUDAGraphRunner):
    # batch=3 pads to 4, then slices back to 3 rows.
    _, _, eager, graph = _run(runner, batch=3, length=16)
    assert graph.shape[0] == 3
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)


def test_large_batch_short_length(runner: BucketedCUDAGraphRunner):
    # batch=8 is only captured for lengths <= large_length_max (16).
    _, _, eager, graph = _run(runner, batch=8, length=16)
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)


def test_eager_fallback_unseen_length(runner: BucketedCUDAGraphRunner):
    # length=10 is not a rung -> chunked eager fallback (still exact).
    _, _, eager, graph = _run(runner, batch=2, length=10)
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)


def test_oversize_batch_chunking(runner: BucketedCUDAGraphRunner):
    # batch=10 > max_pad_size (8) -> chunked into <=8 sub-batches.
    _, _, eager, graph = _run(runner, batch=10, length=8)
    assert graph.shape[0] == 10
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)
