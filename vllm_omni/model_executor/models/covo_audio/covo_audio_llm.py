# Copyright 2026 Tencent.
import os
from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.bucketed_encoder import (
    BucketedCUDAGraphRunner,
    parse_seconds_ladder,
)

logger = init_logger(__name__)

# Whisper mel hop length in samples (WhisperFeatureExtractor default). Bucketed
# waveform rungs (in samples) map to mel frames by integer division.
_WHISPER_HOP_LENGTH = 160
# Env switch for the bucketed encoder path. Unset -> legacy pad-to-30s eager
# path (unchanged behaviour). Set e.g. "0.5,1,2,4,8,16,30" to enable
# length-bucketed padding + bucketed CUDA graphs for the Whisper encoder.
_COVO_AUDIO_BUCKET_SECONDS_ENV = "COVO_AUDIO_BUCKET_SECONDS"
# Large-batch graphs are only captured for short audio (activation ~ bs*T).
_LARGE_CAPTURE_LENGTH_MAX = (8 * 16000) // _WHISPER_HOP_LENGTH  # 8 s -> 800 frames


def _parse_bucketed_ladder():
    """Read the deploy-time bucket ladder, or ``None`` to keep the legacy path."""
    raw = os.environ.get(_COVO_AUDIO_BUCKET_SECONDS_ENV, "").strip()
    if not raw:
        return None
    return parse_seconds_ladder(
        raw,
        sample_rate=16000,
        default_seconds=(),
        min_seconds=0.5,
        max_seconds=30.0,
    )


class DownsampleLayer(nn.Module):
    """
    Downsample layer with 1D convolution and linear layers.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # -> (B, C, T)
        x = self.conv1d(x)  # -> (B, C, T // 2)
        x = x.transpose(1, 2)  # -> (B, T // 2, C)
        x = self.relu1(x)
        x = self.linear1(x)  # -> (B, T // 2, hidden_dim)
        x = self.relu2(x)
        x = self.linear2(x)  # -> (B, T // 2, output_dim)
        return x


class AudioAdapter(nn.Module):
    """
    Audio adapter with downsample layers.
    """

    def __init__(self, input_dim, output_dim, downsample=8):
        """
        Args:
            input_dim (int): input feature dimension (number of channels)
            output_dim (int): output feature dimension
            downsample (int): total downsampling factor, must be a power of 2
        """
        super().__init__()
        assert downsample >= 2 and (downsample & (downsample - 1)) == 0, "downsample must be a power of 2"

        num_layers = (
            downsample.bit_length() - 1
        )  # calculate how many downsampling steps are needed to reach the target factor

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = output_dim if is_last else input_dim
            layers.append(DownsampleLayer(in_dim, out_dim))
            in_dim = out_dim

        self.downsample_layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, T, C),C=input_dim
        Returns:
            Tensor: shape (B, T // downsample, output_dim)
        """
        for layer in self.downsample_layers:
            x = layer(x)
        return x


class CovoAudioLLMForConditionalGeneration(nn.Module, SupportsPP, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.encoder = WhisperEncoder(config.encoder_config)
        self.audio_adapter = AudioAdapter(
            config.whisper_feats_dim, config.llm_config.hidden_size, config.adapter_downsample
        )
        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

        # Bucketed encoder path (deploy-tunable via env var). The ladder is
        # parsed eagerly (fail fast); the CUDA graphs are captured lazily after
        # weights are loaded, when dtype/device are final.
        self._bucketed_ladder = _parse_bucketed_ladder()
        self._bucketed_runner: BucketedCUDAGraphRunner | None = None

    def get_language_model(self) -> torch.nn.Module:
        return self.llm

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def _encode(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Whisper encoder + audio adapter as one graph-capturable forward."""
        feats = self.encoder(audio_features).last_hidden_state
        return self.audio_adapter(feats)

    def _capture_bucketed_graphs(self) -> None:
        """Capture one graph per ``(batch, mel-frames bucket)`` into a shared pool.

        Called from ``load_weights`` once dtype/device are final. A capture
        failure falls back to eager rather than crashing the worker.
        """
        if self._bucketed_ladder is None or self._bucketed_runner is not None:
            return
        try:
            device = next(self.encoder.parameters()).device
            dtype = next(self.encoder.parameters()).dtype
            n_mels = self.encoder.conv1.weight.shape[1]
            length_buckets = sorted({s // _WHISPER_HOP_LENGTH for s in self._bucketed_ladder.buckets})

            def make_inputs(batch: int, length: int) -> tuple[torch.Tensor, ...]:
                return (torch.zeros(batch, length, n_mels, dtype=dtype, device=device),)

            self._bucketed_runner = BucketedCUDAGraphRunner(
                self._encode,
                length_buckets=length_buckets,
                make_inputs=make_inputs,
                capture_sizes=(1, 2, 4, 8),
                large_capture_sizes=(16, 32),
                large_length_max=_LARGE_CAPTURE_LENGTH_MAX,
                eager_chunk=8,
            )
            logger.info("Captured bucketed CUDA graphs for CovoAudio Whisper encoder (frames %s)", length_buckets)
        except Exception:
            logger.warning("Failed to capture bucketed CUDA graphs; falling back to eager", exc_info=True)
            self._bucketed_runner = None

    def _embed_bucketed(
        self,
        audio_features: object,
        audio_num_tokens: object,
        encoder_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Group same-bucket items and run each group through the bucketed graphs.

        ``audio_features`` is either a homogeneous ``(B, T, n_mels)`` tensor or
        a list of per-item ``(T_i, n_mels)`` tensors (mixed-bucket batches).
        Each group is encoded with the matching ``(batch, T)`` graph, then rows
        are trimmed to their real token counts.
        """
        if isinstance(audio_features, (list, tuple)):
            items = [torch.as_tensor(f, dtype=encoder_dtype) for f in audio_features]
        else:
            items = [audio_features[i].to(dtype=encoder_dtype) for i in range(audio_features.size(0))]

        lengths = [int(t) for t in audio_num_tokens] if audio_num_tokens is not None else None
        rows: list[torch.Tensor | None] = [None] * len(items)

        by_t: dict[int, list[int]] = {}
        for i, item in enumerate(items):
            by_t.setdefault(int(item.size(-2)), []).append(i)

        # Graph replay when captured, otherwise eager on the same groups (a
        # capture failure must not take down the worker or the bucketed
        # padding win).
        encode = self._bucketed_runner.execute if self._bucketed_runner is not None else self._encode

        for indices in by_t.values():
            feats_g = torch.stack([items[i] for i in indices], dim=0)
            emb_g = encode(feats_g)
            emb_g = emb_g[0] if self._bucketed_runner is not None else emb_g
            for j, i in enumerate(indices):
                if lengths is not None:
                    rows[i] = emb_g[j, :lengths[i]].clone()
                else:
                    rows[i] = emb_g[j].clone()

        assert all(r is not None for r in rows)
        return rows

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_features = kwargs["audio_features"]
        audio_num_tokens = kwargs.get("audio_num_tokens")
        # Match encoder dtype (bfloat16) since audio_features from WhisperFeatureExtractor are float32
        encoder_dtype = next(self.encoder.parameters()).dtype

        if self._bucketed_ladder is not None:
            return self._embed_bucketed(audio_features, audio_num_tokens, encoder_dtype)

        # Legacy path: pad-to-30s eager encode (unchanged behaviour).
        audio_features = audio_features.to(dtype=encoder_dtype)
        feats = self.encoder(audio_features).last_hidden_state
        features = self.audio_adapter(feats)
        # Truncate each audio to its actual length (Whisper pads all to 30s)
        # Return a list of per-item tensors since lengths may differ.
        batch_size = features.shape[0]
        result = []
        for i in range(batch_size):
            if audio_num_tokens is not None:
                n = int(audio_num_tokens[i])
                result.append(features[i, :n, :])
            else:
                result.append(features[i])
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_state = self.llm(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)
        return hidden_state

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.llm.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        # Capture bucketed CUDA graphs only after weights (dtype/device) are final.
        self._capture_bucketed_graphs()
        return loaded
