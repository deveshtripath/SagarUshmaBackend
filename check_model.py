"""Run from the project folder (venv active):  python check_model.py
Downloads the checkpoint from Hugging Face, loads it with strict layer matching,
and runs random input through it. No Django server needed."""
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from predictor.model_def import CHANNELS, MODALITIES  # noqa: E402
from predictor.model_loader import run  # noqa: E402

T, H, W = 3, 64, 64
inputs = {m: np.random.randn(T, CHANNELS[m], H, W).astype("float32") for m in MODALITIES}
out = run(inputs, 100.0)
print("OK - weights loaded. Output shape:", out.shape)
print("Mean per output channel:", out.mean(axis=(1, 2)))
