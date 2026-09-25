import json
import logging

import numpy as np
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

from .model_def import CHANNELS, MODALITIES, OUTPUT_CHANNELS
from .model_loader import run

log = logging.getLogger(__name__)
MAX_ELEMENTS = 5_000_000


def parse_request(data):
    depth = float(data["depth"])
    raw = data["inputs"]
    arrays, shape = {}, None
    for name in MODALITIES:
        a = np.asarray(raw[name], dtype=np.float32)
        c = CHANNELS[name]
        if a.ndim != 4 or a.shape[1] != c:
            raise ValueError(f"inputs.{name} must have shape [T, {c}, H, W], got {list(a.shape)}")
        this = (a.shape[0], a.shape[2], a.shape[3])
        if shape is None:
            shape = this
        elif this != shape:
            raise ValueError("all inputs must share the same T, H and W")
        if not np.isfinite(a).all():
            raise ValueError(f"inputs.{name} contains NaN or infinity")
        arrays[name] = a
    _, h, w = shape
    if h % 4 or w % 4:
        raise ValueError("H and W must be divisible by 4")
    if sum(a.size for a in arrays.values()) > MAX_ELEMENTS:
        raise ValueError("input too large")
    return arrays, depth


@require_GET
def health(request):
    return JsonResponse({"status": "ok"})


@require_POST
def predict(request):
    try:
        arrays, depth = parse_request(json.loads(request.body))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Body must be valid JSON"}, status=400)
    except KeyError as e:
        return JsonResponse({"error": f"Missing field: {e.args[0]}"}, status=400)
    except (TypeError, ValueError) as e:
        return JsonResponse({"error": f"Invalid input: {e}"}, status=400)

    try:
        y = run(arrays, depth)
    except Exception:
        log.exception("Prediction failed")
        return JsonResponse({"error": "Model error, check server logs"}, status=500)

    return JsonResponse({
        "shape": list(y.shape[1:]),
        "channels": {name: y[i].tolist() for i, name in enumerate(OUTPUT_CHANNELS)},
    })
