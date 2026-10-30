# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for bucketed-encoder length publication (worker side).

Reviewer request: "Add a regression that goes through the real installer and
embed_multimodal..." — these tests drive the REAL ``install_bucketed_encoders``
and the REAL wrapped ``embed_multimodal``; the toy model mirrors the CovoAudio
call convention (``self.encoder(audio_features)`` with no positional lengths)
that commit 9ab317a ("publish lengths to installed bucketed encoder wrappers")
fixes.

* test_real_installer_publishes_lengths_through_embed_multimodal
    lengths published via embed_multimodal must ARRIVE at the installed
    wrapper. Pre-fix the publisher held no wrapper instances (the installer
    assigns the callable to ``module.forward``, which is not in the module
    tree), so ``_lengths`` stayed ``None`` and every call fell back.
* test_cumsum_encoder_runs_bucketed_path_not_fallback
    a position-dependent (cumsum) encoder proves the bucketed path ran and is
    not the eager fallback: per-row prefix equals the solo result, the bucketed
    tail semantics hold, the output width is the largest rung (not raw T), and
    the call replayed captured graphs without re-entering Python forward.
* test_published_lengths_apply_alongside_positional_mask
    call sites that pass ``(features, mask)`` positionally (no lengths
    argument) must still bucket using the published lengths, and stay exact
    for a mask-consuming encoder.
* test_module_pattern_env_override
    ``VLLM_OMNI_AUDIO_BUCKET_MODULES`` selects which module names wrap, so
    encoders outside the built-in name pattern can opt in without model
    changes.

All encoder calls run under ``torch.inference_mode()`` — the production
context: graph capture stores inference tensors, so replay outside
inference_mode raises and the wrapper silently falls back.
"""

import dataclasses

import pytest
import torch
from torch import nn

from vllm_omni.worker.bucketed_encoder import install_bucketed_encoders

pytestmark = [pytest.mark.core_model]

torch.manual_seed(0)

LENGTHS = [40, 120, 200, 75]
T = 240  # feature frames; longer than the largest rung (200)
D = 128
# VLLM_OMNI_AUDIO_BUCKET_SECONDS is parsed with 100 frames per second.
RUNGS = (40, 100, 200)


@dataclasses.dataclass
class EncOutput:
    last_hidden_state: torch.Tensor


class RecordingEncoder(nn.Module):
    """cumsum encoder: output[t] = sum_{s<=t} x[s] (position-dependent)."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def forward(self, x: torch.Tensor) -> EncOutput:
        self.calls.append(tuple(x.shape))
        return EncOutput(last_hidden_state=x.cumsum(dim=1))


class ToyModel(nn.Module):
    """CovoAudio-style embed_multimodal: encoder called without lengths."""

    def __init__(self):
        super().__init__()
        self.encoder = RecordingEncoder()

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        audio_features = kwargs["audio_features"]
        return self.encoder(audio_features).last_hidden_state


def make_features(lengths=LENGTHS, t: int = T, dim: int = D) -> torch.Tensor:
    """Rows carry real data before their true length and garbage after."""
    feats = torch.randn(len(lengths), t, dim, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, length:] += 3.0  # garbage tail: makes fallback outputs differ
    return feats


def rung_of(length: int) -> int:
    for rung in RUNGS:
        if rung >= length:
            return rung
    return RUNGS[-1]


def expected_bucketed(feats: torch.Tensor, lengths=LENGTHS) -> torch.Tensor:
    """The wrapper contract: per row cumsum over (true prefix + zero pad to
    its rung), scattered into a container as wide as the largest rung used;
    rows with a shorter rung stay zero beyond it."""
    width = max(rung_of(length) for length in lengths)
    out = torch.zeros(len(lengths), width, D, device="cuda")
    for row, length in enumerate(lengths):
        rung = rung_of(length)
        window = torch.zeros(rung, D, device="cuda")
        window[:length] = feats[row, :length]
        out[row, :rung] = window.cumsum(dim=0)
    return out


@pytest.fixture()
def bucket_env(monkeypatch):
    # Real env names from the code (not the drifted names in the commit message).
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "0.4,1,2")
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_BATCH_SIZES", "1,2,4")
    if not torch.cuda.is_available():
        pytest.skip("bucketed encoder requires CUDA")


def test_real_installer_publishes_lengths_through_embed_multimodal(bucket_env):
    from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor

    model = ToyModel().cuda()
    wrapped = install_bucketed_encoders(model)
    assert wrapped == ["encoder"], f"unexpected wrapped modules: {wrapped}"
    assert getattr(OmniMultiModalProcessor, "_bucket_frames_installed", False)

    encoder_wrapper = model.encoder.forward  # installer assigns the _BucketedEncoder here
    assert type(encoder_wrapper).__name__ == "_BucketedEncoder"
    assert encoder_wrapper._lengths is None  # nothing published yet

    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )

    # Core regression assertion: the published lengths ARRIVE at the wrapper.
    assert encoder_wrapper._lengths == tuple(LENGTHS)

    # Functional echo: a follow-up call without positional lengths is grouped
    # by the published lengths -> per-rung padded executions hit the pool.
    assert encoder_wrapper.pool is not None, "graph pool was not captured"
    seen: list[tuple] = []
    original_execute = encoder_wrapper.pool.execute

    def counting_execute(*args, **kwargs):
        seen.append(tuple(args[0].shape))
        return original_execute(*args, **kwargs)

    encoder_wrapper.pool.execute = counting_execute
    calls_before = len(model.encoder.calls)
    with torch.inference_mode():
        model.encoder(feats)
    # Grouping evidence: one padded execution per length rung present.
    assert sorted(seen) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), seen
    # Replay does not re-enter Python forward; a fallback would have.
    assert len(model.encoder.calls) == calls_before


def test_cumsum_encoder_runs_bucketed_path_not_fallback(bucket_env):
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    with torch.inference_mode():
        # First eligible call: eager (learns output type + captures the pool).
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        calls_after_prime = len(model.encoder.calls)
        out = model.encoder(feats).last_hidden_state
        calls_after_bucketed = len(model.encoder.calls)

    expected = expected_bucketed(feats)

    # (a) Position-dependent proof, per row: the cumsum signature of each row's
    #     TRUE prefix must land in its own row (scatter is row-aligned) ...
    for row, length in enumerate(LENGTHS):
        solo_prefix = feats[row, :length].cumsum(dim=0)
        torch.testing.assert_close(out[row, :length], solo_prefix, rtol=0, atol=1e-6)

    # ... and the bucketed tail semantics hold: flat to the row's rung, zeros after.
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)

    # (b) Bucketed output is structurally DIFFERENT from the fallback's:
    #     width = largest rung (200), not the raw T (240).
    assert out.shape == (len(LENGTHS), 200, D), out.shape
    # The fallback computes cumsum over each full garbage tail; its values in
    # the tail region differ from the bucketed (zero-padded) ones.
    fallback_tail = feats.cumsum(dim=1)[:, 40:200]
    assert not torch.allclose(out[0, 40:200], fallback_tail[0], atol=1e-3)

    # (c) Execution evidence: the run happened as per-rung graph replays
    #     (checked before any solo runs below that call the original forward).
    assert wrapper.pool is not None and wrapper.pool.entries, "no captured graphs"
    for key in ((1, 40), (2, 200), (1, 100)):
        assert key in wrapper.pool.entries, f"missing captured entry {key}"
    assert calls_after_bucketed == calls_after_prime, (
        "Python forward re-entered during the bucketed call (fallback path)"
    )


class MaskConsumingEncoder(nn.Module):
    """Mask-aware encoder: output depends on both features and the mask."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).cumsum(dim=1) + w.cumsum(dim=1)


class MaskToyModel(nn.Module):
    """Embed path that calls the encoder as ``encoder(feats, mask)``."""

    def __init__(self):
        super().__init__()
        self.encoder = MaskConsumingEncoder()

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        return self.encoder(kwargs["audio_features"], kwargs["feature_attention_mask"])


def test_published_lengths_apply_alongside_positional_mask(bucket_env):
    """(features, mask) call sites must bucket via the published lengths and
    stay numerically exact for a mask-consuming encoder."""
    model = MaskToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    mask = torch.zeros(len(LENGTHS), T, dtype=torch.int32, device="cuda")
    for row, length in enumerate(LENGTHS):
        mask[row, :length] = 1

    with torch.inference_mode():
        model.embed_multimodal(
            audio_features=feats,
            feature_attention_mask=mask,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        assert wrapper._lengths == tuple(LENGTHS)
        out = model.encoder(feats, mask)

    assert out.shape == (len(LENGTHS), 200, D), out.shape
    for row, length in enumerate(LENGTHS):
        solo = wrapper._original(feats[row : row + 1, :length], mask[row : row + 1, :length])[0]
        torch.testing.assert_close(out[row, :length], solo, rtol=0, atol=1e-6)


def test_module_pattern_env_override(bucket_env, monkeypatch):
    """VLLM_OMNI_AUDIO_BUCKET_MODULES selects which module names wrap."""
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_MODULES", r"(?:^|\.)my_frontend$")

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_frontend = RecordingEncoder()
            self.encoder = RecordingEncoder()

        def embed_multimodal(self, **kwargs):
            return self.my_frontend(kwargs["audio_features"]).last_hidden_state

    model = CustomModel().cuda()
    assert install_bucketed_encoders(model) == ["my_frontend"]


def test_fixed_window_encoder_contract_engages(bucket_env, monkeypatch):
    """The tencent/Covo-Audio-Chat call contract (community model, added in
    PR #2293): a single positional ``(B, mel, T)`` input to a fixed-window HF
    WhisperEncoder, lengths published via embed_multimodal. The module's
    time-axis handling and fixed-window recipe must engage the bucketed path
    with zero model-side changes: fixed-window forward untouched, windows
    captured per rung, and exact output where rung == true length."""
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "3,8,30")  # rungs 300/800/3000
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_BATCH_SIZES", "1,2,4")
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    raw_calls = []
    original_forward = WhisperEncoder.forward

    def counting_forward(self, *args, **kwargs):
        raw_calls.append(tuple(args[0].shape))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(WhisperEncoder, "forward", counting_forward)

    class CovoContractModel(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = WhisperConfig(d_model=64, encoder_layers=1, encoder_attention_heads=2,
                                encoder_ffn_dim=128, num_mel_bins=80, max_source_positions=1500)
            self.encoder = WhisperEncoder(cfg).cuda().eval()

        def embed_multimodal(self, **kwargs):  # covo_audio_llm.py:116 contract
            return self.encoder(kwargs["audio_features"]).last_hidden_state

    model = CovoContractModel()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    lengths = [300, 800, 2400]
    feats = torch.randn(len(lengths), 80, 3000, device="cuda")  # (B, mel, T) padded to 30s
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(lengths))
        assert wrapper._lengths == tuple(lengths)
        out = model.encoder(feats)

    # Engaged: per-rung windows were captured and replayed (fallback never
    # captures a 300/800-frame window, and calls the fixed-window forward).
    assert wrapper.pool is not None and wrapper.pool.entries
    for key in ((1, 300), (1, 800), (1, 3000)):
        assert key in wrapper.pool.entries, f"missing captured entry {key}"
    assert raw_calls == [], f"fixed-window forward was called: {raw_calls}"

    # Exact where the rung equals the true length (no padding involved).
    hidden = out.last_hidden_state
    assert hidden.shape[0] == len(lengths) and hidden.shape[2] == 64
    solo_row0 = wrapper._original(feats[0:1, :, :300])
    torch.testing.assert_close(hidden[0, :150], solo_row0.last_hidden_state[0], rtol=0, atol=1e-5)
