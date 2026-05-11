from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import numpy as np


def encode_action(action: Mapping[str, Any], source_id: str, stamp: float | None = None) -> str:
    return json.dumps(
        {
            "source_id": source_id,
            "stamp": _stamp(stamp),
            "action": _to_json_value(dict(action)),
        },
        separators=(",", ":"),
    )


def decode_action(msg: str | Any) -> dict[str, Any]:
    payload = _loads(msg)
    if not isinstance(payload.get("action"), dict):
        raise ValueError("Action payload must contain an 'action' object")
    if not isinstance(payload.get("source_id"), str) or not payload["source_id"]:
        raise ValueError("Action payload must contain a non-empty 'source_id'")
    if not isinstance(payload.get("stamp"), int | float):
        raise ValueError("Action payload must contain a numeric 'stamp'")
    return payload


def encode_observation(
    observation: Mapping[str, Any], stamp: float | None = None, exclude_images: bool = True
) -> str:
    values = {
        key: _to_json_value(value)
        for key, value in observation.items()
        if not (exclude_images and _is_image_value(value))
    }
    return json.dumps({"stamp": _stamp(stamp), "observation": values}, separators=(",", ":"))


def decode_observation(msg: str | Any) -> dict[str, Any]:
    payload = _loads(msg)
    if not isinstance(payload.get("observation"), dict):
        raise ValueError("Observation payload must contain an 'observation' object")
    if not isinstance(payload.get("stamp"), int | float):
        raise ValueError("Observation payload must contain a numeric 'stamp'")
    return payload


def _loads(msg: str | Any) -> dict[str, Any]:
    data = msg.data if hasattr(msg, "data") else msg
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _stamp(stamp: float | None) -> float:
    return time.time() if stamp is None else float(stamp)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_value(item) for item in value]
    return value


def _is_image_value(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim in (2, 3)
