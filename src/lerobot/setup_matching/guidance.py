#!/usr/bin/env python

from __future__ import annotations


def movement_guidance(
    label: str, target_centroid: tuple[float, float], live_centroid: tuple[float, float]
) -> str:
    delta_x = target_centroid[0] - live_centroid[0]
    delta_y = target_centroid[1] - live_centroid[1]
    horizontal = _horizontal_direction(delta_x)
    vertical = _vertical_direction(delta_y)
    directions = [direction for direction in (horizontal, vertical) if direction]
    if not directions:
        return f"{label}: hold position"
    return f"{label}: move {' and '.join(directions)}"


def _horizontal_direction(delta_x: float, deadband: float = 0.01) -> str | None:
    if abs(delta_x) <= deadband:
        return None
    return "right" if delta_x > 0 else "left"


def _vertical_direction(delta_y: float, deadband: float = 0.01) -> str | None:
    if abs(delta_y) <= deadband:
        return None
    return "down" if delta_y > 0 else "up"
