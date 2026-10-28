# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical-equivalence tests for the CovoAudio bucketed encoder path.

Exercises the reference integration of
:class:`vllm_omni.model_executor.bucketed_encoder.BucketedCUDAGraphRunner`
against the real CovoAudio encoder structure — a HuggingFace
``WhisperEncoder`` followed by the CovoAudio ``AudioAdapter`` (three
stride-2 downsample layers) — plus the same-bucket grouping/trim logic in
``CovoAudioLLMForConditionalGeneration._embed_bucketed``.

The encoder is independent across batch rows (Whisper self-attention within
each row only), so graph replay must match eager up to floating-point
tolerance. A tiny Whisper config keeps the test fast while preserving the
stride-2 downsample chain that makes lengths bucket-sensitive.
"""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.bucketed_encoder import BucketedCUDAGraphRunner
from vllm_omni.model_executor.models.covo_audio.covo_audio_llm import (
    AudioAdapter,
    CovoAudioLLMForConditionalGeneration,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}),
]

DEVICE = torch.device("cuda:0")
N_MELS = 16
D_MODEL = 32
OUT_DIM = 32


@pytest.fixture(scope="module")
def encode():
    """WhisperEncoder + AudioAdapter in one forward (the graph-capture unit)."""
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    torch.manual_seed(0)
    cfg = WhisperConfig(
        num_mel_bins=N_MELS,
        d_model=D_MODEL,
        encoder_layers=2,
        encoder_attention_heads=4,
        max_source_positions=64,
    )
    encoder = WhisperEncoder(cfg).to(DEVICE)
    adapter = AudioAdapter(D_MODEL, OUT_DIM, downsample=8).to(DEVICE)

    def _encode(features: torch.Tensor) -> torch.Tensor:
        feats = encoder(features).last_hidden_state
        return adapter(feats)

    return _encode


@pytest.fixture(scope="module")
def runner(encode) -> BucketedCUDAGraphRunner:
    def make_inputs(batch: int, length: int) -> tuple[torch.Tensor, ...]:
        return (torch.randn(batch, length, N_MELS, device=DEVICE),)

    return BucketedCUDAGraphRunner(
        encode,
        length_buckets=(16, 32),
        make_inputs=make_inputs,
        capture_sizes=(1, 2, 4),
    )


def test_encode_graph_equivalence(encode, runner: BucketedCUDAGraphRunner):
    features = torch.randn(2, 16, N_MELS, device=DEVICE)
    with torch.inference_mode():
        eager = encode(features)
        (graph,) = runner.execute(features)
    torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-5)


def test_embed_bucketed_grouping(encode, runner: BucketedCUDAGraphRunner):
    # Mixed-bucket batch: two items at T=16 (1 output token), one at T=32 (2).
    items = [
        torch.randn(16, N_MELS, device=DEVICE),
        torch.randn(32, N_MELS, device=DEVICE),
        torch.randn(16, N_MELS, device=DEVICE),
    ]
    audio_num_tokens = torch.tensor([1, 2, 1], dtype=torch.int64)

    # Drive the real model method without constructing the full model (needs a
    # vllm_config / loaded LLM); _embed_bucketed only depends on _bucketed_runner.
    model = CovoAudioLLMForConditionalGeneration.__new__(CovoAudioLLMForConditionalGeneration)
    model._bucketed_runner = runner

    with torch.inference_mode():
        rows = model._embed_bucketed(items, audio_num_tokens, torch.float32)

    # Reference: eager encode per item, trimmed to its real token count.
    with torch.inference_mode():
        refs = [
            encode(item.unsqueeze(0))[0, :int(audio_num_tokens[i])]
            for i, item in enumerate(items)
        ]

    assert len(rows) == len(items)
    for got, ref in zip(rows, refs):
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
