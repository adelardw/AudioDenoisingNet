import torchaudio

from lightning_modules.lightning_module import UltraSpectrogramLightningModelUnet
from utils.edge_fix import run_fixed

ckpt_path = 'ckpts/ultra/checkpoints/last.ckpt'
input_audio_path = 'examples/example.wav'
save_cleaned_audio_path = 'denoised_example.wav'

model = UltraSpectrogramLightningModelUnet.load_from_checkpoint(ckpt_path, map_location='cpu').eval()
audio, rate = torchaudio.load(input_audio_path)  # (channels, samples)
audio = torchaudio.functional.resample(audio, rate, 16000)  # the model works at 16 kHz
denoised = run_fixed(model, audio.mean(0))  # mono in, same length out, no edge clicks
torchaudio.save(save_cleaned_audio_path, denoised.unsqueeze(0), 16000)
