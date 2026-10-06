"""Inference-time fix for the ISTFT edge artifact (no retraining required).

The STFT is configured with ``n_fft=2046``, ``win_length=256``, ``hop_length=123`` and
``center=False``. The 256-sample window sits in the middle of every 2046-sample frame, so the
first and last ~900 samples (~56 ms) of each processed chunk are never covered by an analysis
window. Right at the coverage boundary the inverse STFT divides by a near-zero window envelope,
so the edges of every processed chunk are zeroed and a loud click appears at 0.056 s. ``model.run``
also splits audio longer than 8 s into independent 8 s chunks, so this repeats every 8 s.

``run_fixed`` reflect-pads the input so the windows fully cover the real signal, runs the model
once on a length the U-Net accepts (STFT frame counts of the form ``256 * j + 1``; 128000 samples
= 8 s is the training length) and trims the padding back off. Clips up to ~7.7 s cost exactly the
same as ``model.run``; longer clips are processed in a single pass instead of 8 s chunks.

Usage::

    model = UltraSpectrogramLightningModelUnet.load_from_checkpoint(ckpt, map_location="cpu").eval()
    cleaned = run_fixed(model, noisy)  # noisy: mono 16 kHz waveform, shape (T,), (1, T) or (1, 1, T)
"""
import torch
import torch.nn.functional as F

__all__ = ["run_fixed"]


@torch.no_grad()
def run_fixed(model, x, pad=2048, n_fft=2046, hop_length=123):
    """Denoise a mono waveform without edge artifacts. Returns a 1-D tensor of the same length."""
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.numel() != x.shape[-1]:
        raise ValueError("run_fixed expects mono audio; downmix first, e.g. audio.mean(0)")
    length = x.shape[-1]
    pad = min(pad, length - 1)  # reflect padding must be shorter than the signal
    padded = F.pad(x.reshape(1, 1, -1), (pad, pad), mode="reflect").reshape(1, -1)

    blocks = 4  # n_fft + hop_length * 256 * 4 = 128000 samples (training length)
    while n_fft + hop_length * 256 * blocks < padded.shape[-1]:
        blocks += 1

    original_len = model.audio_len
    model.audio_len = max(original_len, n_fft + hop_length * 256 * blocks)
    try:
        y = model.run(padded)
    finally:
        model.audio_len = original_len
    return y.reshape(-1)[pad:pad + length]
