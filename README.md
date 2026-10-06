# AudioDenoisingNet

Lightweight CNN-based speech enhancement (audio denoising) in the spectrogram domain. The model takes the magnitude spectrogram of noisy audio, reconstructs a clean magnitude with a U-Net, and recovers phase with a small dedicated phase-reconstruction head. It is designed to favour **inference speed** (telephony, mobile, IoT, ASR front-ends) over raw SOTA quality, while still staying competitive on standard speech-enhancement benchmarks.

This repository contains the code accompanying the master's thesis *"Investigation of deep-neural-network-based denoising methods for audio signals"* (HSE University, Applied Mathematics and Informatics / Machine Learning and Data Analysis).

## Key ideas

- **STFT magnitude in, clean magnitude out.** Audio is transformed with a Short-Time Fourier Transform (Hann window). A U-Net encoder–decoder with skip connections cleans the magnitude spectrogram, and the signal is reconstructed with the inverse STFT.
- **Explicit phase handling.** A small (~83K-parameter) phase-reconstruction head takes the noisy phase together with the cleaned magnitude and predicts a clean phase. This removes the negative-SNR artifact you get from naively reusing the noisy phase.
- **Composite loss.** Training combines magnitude losses (L1/L2), Spectral Convergence Loss, Phase Sensitive Loss, Group Delay Loss, a plain phase loss, and an STFT loss. Loss-term weights `α`, `β` are learnable.
- **~2M parameters total.** The whole pipeline (1.9M U-Net + ~83K phase head) stays under 2M parameters.

## Results

Evaluated on **VoiceBank + DEMAND** and **LibriSpeech + WHAM!**.

| Metric | VoiceBank + DEMAND | LibriSpeech + WHAM! |
|---|---|---|
| SNR | 3.84 dB | 8.23 dB |
| SDR | 5.40 dB | 7.96 dB |
| SI-SDR | 5.20 dB | 7.65 dB |
| SI-SNR | 5.20 dB | 7.66 dB |
| STOI | 0.902 | 0.900 |
| PESQ-NB | 2.699 | 2.426 |
| PESQ-WB | 1.876 | 1.784 |

> These are the thesis numbers, measured with the original inference path. It has an edge artifact that lowers every metric; with the fix, the same checkpoint reaches SI-SDR 9.54 dB, PESQ-WB 2.014, PESQ-NB 2.907 and STOI 0.913 on VoiceBank + DEMAND (see [Edge-artifact fix](#edge-artifact-fix)).

Inference speed (single audio clip, Nvidia Tesla T4, VoiceBank + DEMAND):

| Model | Params | Time |
|---|---|---|
| DCCRN | 3M+ | 0.0106 s |
| MP-SEUnet | 2.2M | 0.0226 s |
| TF-Locoformer | 7.8M | 0.0240 s |
| **AudioDenoisingNet (GPU)** | **< 2M** | **0.0047 s** |
| **AudioDenoisingNet (GPU + CPU)** | **< 2M** | **0.065 s** |

Quality is below transformer/Mamba SOTA models (e.g. TF-Locoformer reaches SI-SDR > 15 dB), but this model is the fastest on GPU in the comparison above, which is the trade-off it is built for. The STFT/ISTFT and padding/reshape steps run on CPU and dominate the CPU-side latency.

## Edge-artifact fix

The STFT uses `n_fft=2046`, `win_length=256`, `hop_length=123` and `center=False`, so the 256-sample window covers only the middle of each 2046-sample frame. The first and last ~900 samples (~56 ms) of every processed chunk are never covered by a window, and at the coverage boundary the inverse STFT divides by a near-zero window envelope. With `model.run`, the first ~56 ms of every output are zeroed and followed by a loud click at 0.056 s. Audio longer than 8 s is processed as independent 8 s chunks, so this repeats every 8 s.

`utils/edge_fix.py` fixes this at inference time, with no retraining: `run_fixed` reflect-pads the input by 2048 samples, runs the network once on a length the U-Net accepts (STFT frame counts of the form `256 * j + 1`) and trims the padding. For clips up to ~7.7 s the network sees the same 8 s window as before, so the cost does not change.

Full VoiceBank + DEMAND test set (824 clips, 16 kHz), checkpoint `ckpts/ultra/checkpoints/last.ckpt`:

| | SI-SDR, dB | PESQ-WB | PESQ-NB | STOI |
|---|---|---|---|---|
| Noisy input (no processing) | 8.45 | 1.971 | 2.880 | 0.921 |
| Original inference (`model.run`) | 5.20 | 1.876 | 2.699 | 0.902 |
| **Fixed inference (`run_fixed`)** | **9.54** | **2.014** | **2.907** | **0.913** |

Paired differences over the same 824 clips (mean ± 95% CI):

| | SI-SDR, dB | PESQ-WB | PESQ-NB | STOI |
|---|---|---|---|---|
| Fixed vs. original inference | +4.34 ± 0.30 | +0.139 ± 0.017 | +0.208 ± 0.025 | +0.011 ± 0.001 |
| Fixed vs. noisy input | +1.10 ± 0.14 | +0.044 ± 0.022 | +0.027 ± 0.015 | -0.008 ± 0.001 |

The original-inference row reproduces the thesis numbers exactly; because of the artifact it is below the unprocessed input on every metric. With the fix, the model improves on the noisy input in SI-SDR and PESQ, while STOI stays slightly below it. LibriSpeech + WHAM! has not been re-evaluated yet.

To reproduce (the test split is downloaded from the Hugging Face Hub):

```bash
python benchmarks/eval_vbd_edge_fix.py --ckpt ckpts/ultra/checkpoints/last.ckpt --out vbd_edge_fix.csv
```

## Repository structure

```
AudioDenoisingNet/
├── configs/             # YAML model/training configs (e.g. denoise_model_v1_cfg.yaml)
├── lightning_modules/   # PyTorch Lightning modules (model + train/val/test steps)
├── loaders/             # Dataset and DataLoader construction (get_loaders)
├── utils/               # Helpers, incl. config loader (cfg_loader.load_cfg) and edge_fix.run_fixed
├── benchmarks/          # VoiceBank + DEMAND evaluation of the inference edge-artifact fix
├── onnx_converts/       # ONNX export scripts
├── examples/            # Example audio / usage
├── ckpts/               # Checkpoints
├── train.py             # Training entrypoint
├── eval.py              # Inference / denoising entrypoint
├── test_model.ipynb     # Exploration / evaluation notebook
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/adelardw/AudioDenoisingNet.git
cd AudioDenoisingNet
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchaudio`, `librosa`, `lightning`.

## Data

Training uses **LibriSpeech** (clean speech) mixed with **WHAM!** (noise). Expected layout:

```
dataset/
├── dev-clean/                  # LibriSpeech clean speech
├── test-clean.tar              # LibriSpeech test split
└── wham_noise/wham_noise/      # WHAM! noise
```

- LibriSpeech: https://www.openslr.org/12
- WHAM!: https://wham.whisper.ai/
- VoiceBank + DEMAND (benchmark): https://datashare.ed.ac.uk/handle/10283/2791

## Usage

### Training

Point `train.py` at a config and your dataset directories, then run:

```bash
python train.py
```

Internally this looks like:

```python
from lightning_modules.lightning_module import *
from loaders import *
from utils import *
import lightning as L

cfg_path = 'configs/denoise_model_v1_cfg.yaml'

train_loader, val_loader, test_loader = get_loaders(
    speech_dirs=["dataset/dev-clean", "dataset/test-clean.tar"],
    noise_dir="dataset/wham_noise/wham_noise",
    batch_size=16,
    padding_strategy=None,
)

model = UltraSpectrogramLightningModelUnet(**load_cfg(cfg_path))

trainer = L.Trainer(accelerator="auto", max_epochs=300, logger=True)
trainer.fit(model, train_loader, val_loader)
```

> The repo contains several Lightning model variants (e.g. `SpectrogramLightningModelUnet`, `UltraSpectrogramLightningModelUnet`, `GiGaSpectrogramLightningModelUnet`). Pick the one matching your config; you can also warm-start one model from another's `state_dict`.

### Inference / denoising a file

```python
import torchaudio
from lightning_modules.lightning_module import UltraSpectrogramLightningModelUnet
from utils.edge_fix import run_fixed

model = UltraSpectrogramLightningModelUnet.load_from_checkpoint(
    'ckpts/ultra/checkpoints/last.ckpt', map_location='cpu'
).eval()

audio, rate = torchaudio.load('examples/example.wav')       # (channels, samples)
audio = torchaudio.functional.resample(audio, rate, 16000)  # the model works at 16 kHz
denoised = run_fixed(model, audio.mean(0))                  # mono in, same length out
torchaudio.save('denoised_example.wav', denoised.unsqueeze(0), 16000)
```

Or just run the provided script:

```bash
python eval.py
```

`run_fixed` processes a clip of any length in one pass. Calling `model.run(audio)` directly still works, but it leaves the edge artifact described in [Edge-artifact fix](#edge-artifact-fix), and on audio longer than 8 s the artifact repeats every 8 s.

## Training configuration

Final model (phase-aware), summarized from the thesis:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| LR (U-Net 1.9M) | 0.00341 |
| LR (phase head 83K) | 0.0001 |
| Weight decay | 0.2 |
| Scheduler | Linear warmup (start factor 0.01, 5 iters) → Cosine Annealing (T_max 25, eta_min 5e-5) |
| Epochs | 50 |
| Batch size | 16 |
| Bottleneck activation | Tanh |
| Losses | L1, L2, Phase Sensitive, Spectral Convergence, Group Delay, Phase, STFT |
| Total params | ~2M |

## Limitations & roadmap

- Quality trails attention/SSM-based SOTA models; the current phase handling does not fully remove phase distortion.
- CPU inference is dominated by STFT/ISTFT and tensor reshaping.
- Planned: attention/Transformer and Mamba (SSM) blocks for higher quality, and further optimization for mobile devices. For telephony, add compression/channel-distortion augmentations during training.

## Practical applications

- Cleaning voice messages in messengers (the model is trained on real-world street noise).
- Phone-call enhancement (with appropriate telephony augmentations added at training time).
- A denoising front-end in an ASR pipeline.

## Citation

If this is useful, please cite the thesis:

> Sergaev Y. S. *Investigation of deep-neural-network-based denoising methods for audio signals.* Master's thesis, HSE University, School of Computer Science, Physics and Technology.

## License

No license file is currently present in the repository. Add one (e.g. MIT) if you intend others to reuse the code.
