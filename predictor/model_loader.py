"""Downloads the model from Hugging Face once, keeps it in memory, and runs predictions."""
import os
import threading

import torch
from huggingface_hub import hf_hub_download

from .model_def import build_model, finish_output, prepare_depth, prepare_input

_lock = threading.Lock()
_cache = {}


def _load():
    path = hf_hub_download(
        repo_id=os.environ["HF_REPO_ID"],
        filename=os.environ["HF_MODEL_FILENAME"],
        token=os.environ.get("HF_TOKEN"),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    state = {k.removeprefix("module."): v for k, v in state.items()}

    model = build_model()
    model.load_state_dict(state)  # strict: every layer name and shape must match
    model.eval()
    return model


def _get():
    if "model" not in _cache:
        with _lock:
            if "model" not in _cache:
                _cache["model"] = _load()  # first request downloads + loads (slow once)
    return _cache["model"]


def run(inputs, depth):
    """inputs: {name: numpy (T, C, H, W)}, depth: float -> numpy (2, H, W)"""
    model = _get()
    x = {
        k: torch.from_numpy(prepare_input(k, v)).float().unsqueeze(0)
        for k, v in inputs.items()
    }
    d = torch.tensor([prepare_depth(depth)], dtype=torch.float32)
    with torch.inference_mode():
        y = model(x, d)[0].numpy()
    return finish_output(y)
