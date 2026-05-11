#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from xml.etree import ElementTree

import cv2
import numpy as np

from .artifacts import ArtifactStore, EpisodeReviewStore
from .detectors import TemplateMatchingDetector, create_object_template
from .frame_sources import EpisodeReplayFrameSource
from .models import BoundingBox, MatchResult, ReferenceState, SetupProfile
from .scenario import SetupMatchScenario, _camera_drift_warnings, run_setup_match_scenario
from .scoring import ScoreConfig, score_setup

DEFAULT_FAILED_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/"
    "rollout_1_so101_green_lego_in_metal_holder_20260511_114146"
)
DEFAULT_MOCK_ROBOT_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/so101_green_lego_in_metal_holder"
)
DEFAULT_CAMERA_KEY = "observation.images.side"
DEFAULT_ARTIFACT_ROOT = Path("outputs/pre_record_setup_matching_web")
_FAILURE_CATEGORIES = ["missed_pick", "wrong_place", "collision", "dropped_object", "other"]
_STATIC_ROOT = Path(__file__).with_name("static")


class SetupMatchingRequestHandler(SimpleHTTPRequestHandler):
    artifact_root = DEFAULT_ARTIFACT_ROOT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(_STATIC_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            self._send_json(
                {
                    "failed_dataset_path": str(DEFAULT_FAILED_DATASET),
                    "mock_robot_dataset_path": str(DEFAULT_MOCK_ROBOT_DATASET),
                    "artifact_root": str(self.artifact_root),
                    "camera_key": DEFAULT_CAMERA_KEY,
                    "reference_episode_index": 0,
                    "reference_frame_index": 0,
                    "replay_episode_index": 0,
                    "failure_categories": _FAILURE_CATEGORIES,
                }
            )
            return
        if parsed.path == "/api/dataset":
            query = parse_qs(parsed.query)
            dataset_path = Path(_first_query_value(query, "path", str(DEFAULT_FAILED_DATASET)))
            self._send_json(_dataset_info(dataset_path, self.artifact_root))
            return
        if parsed.path == "/api/frame":
            query = parse_qs(parsed.query)
            frame = _read_episode_frame(
                Path(_first_query_value(query, "dataset_path", str(DEFAULT_FAILED_DATASET))),
                _first_query_value(query, "camera_key", DEFAULT_CAMERA_KEY),
                int(_first_query_value(query, "episode_index", "0")),
                int(_first_query_value(query, "frame_index", "0")),
            )
            self._send_json(_frame_payload(frame))
            return
        if parsed.path == "/api/joints":
            query = parse_qs(parsed.query)
            self._send_json(_joint_comparison_from_query(query))
            return
        if parsed.path == "/ws/match" and self.headers.get("Upgrade", "").lower() == "websocket":
            self._handle_match_websocket()
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/reviews":
                self._send_json(self._save_review(payload))
                return
            if parsed.path == "/api/refine-roi":
                self._send_json(self._refine_roi(payload))
                return
            if parsed.path == "/api/match":
                self._send_json(self._run_match(payload))
                return
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except Exception as exc:  # noqa: BLE001 - local prototype should surface errors to the browser.
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_path = Path(str(payload.get("dataset_path", DEFAULT_FAILED_DATASET)))
        episode_index = int(payload.get("episode_index", 0))
        status = str(payload.get("status", "unreviewed"))
        if status not in ("failed", "good", "skipped", "unreviewed"):
            raise ValueError(f"Unsupported review status: {status}")
        store = EpisodeReviewStore(self.artifact_root / "episode_reviews.json")
        store.set_review(
            dataset_path,
            episode_index,
            status,  # type: ignore[arg-type]
            failure_category=_optional_str(payload.get("failure_category")),
            note=_optional_str(payload.get("note")),
        )
        return {
            "ok": True,
            "review": store.get_review(dataset_path, episode_index),
            "dataset": _dataset_info(dataset_path, self.artifact_root),
        }

    def _refine_roi(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = _read_episode_frame(
            Path(str(payload.get("dataset_path", DEFAULT_FAILED_DATASET))),
            str(payload.get("camera_key", DEFAULT_CAMERA_KEY)),
            int(payload.get("episode_index", 0)),
            int(payload.get("frame_index", 0)),
        )
        bbox = _bbox_from_mapping(payload.get("bbox", {}))
        refined_bbox = _refine_bbox_with_color_blob(frame, bbox)
        return {"bbox": asdict(refined_bbox)}

    def _run_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario = _scenario_from_payload(payload)
        artifact_store = ArtifactStore(self.artifact_root) if bool(payload.get("save", True)) else None
        result = run_setup_match_scenario(scenario, artifact_store=artifact_store).to_dict()
        result["artifact_root"] = str(self.artifact_root)
        return result

    def _handle_match_websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._send_error(HTTPStatus.BAD_REQUEST, "Missing Sec-WebSocket-Key")
            return
        accept_key = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key)
        self.end_headers()

        try:
            config = json.loads(_websocket_read_text(self.rfile))
            for payload in _stream_match_payloads(config, self.artifact_root):
                _websocket_write_text(self.wfile, json.dumps(payload))
                time.sleep(float(config.get("interval_seconds", 0.12)))
            _websocket_write_text(self.wfile, json.dumps({"type": "complete"}))
        except Exception as exc:  # noqa: BLE001
            _websocket_write_text(self.wfile, json.dumps({"type": "error", "message": str(exc)}))

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)


def run_server(host: str, port: int, artifact_root: Path) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredSetupMatchingRequestHandler",
        (SetupMatchingRequestHandler,),
        {"artifact_root": artifact_root},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Pre-record setup matching UI: http://{host}:{server.server_port}")
    print(f"Artifacts: {artifact_root}")
    server.serve_forever()
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local pre-record setup matching PoC web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_server(args.host, args.port, args.artifact_root)
    return 0


def _scenario_from_payload(payload: dict[str, Any]) -> SetupMatchScenario:
    return SetupMatchScenario(
        reference_dataset_path=Path(str(payload.get("reference_dataset_path", DEFAULT_FAILED_DATASET))),
        replay_dataset_path=Path(str(payload.get("replay_dataset_path", DEFAULT_MOCK_ROBOT_DATASET))),
        camera_key=str(payload.get("camera_key", DEFAULT_CAMERA_KEY)),
        reference_episode_index=int(payload.get("reference_episode_index", 0)),
        reference_frame_index=int(payload.get("reference_frame_index", 0)),
        replay_episode_index=int(payload.get("replay_episode_index", 0)),
        replay_start_frame_index=int(payload.get("replay_start_frame_index", 0)),
        object_rois=_object_rois_from_payload(payload),
        stream_frames=int(payload.get("stream_frames", 1)),
    )


def _object_rois_from_payload(payload: dict[str, Any]) -> dict[str, BoundingBox] | None:
    raw_objects = payload.get("object_rois") or payload.get("objects")
    if not raw_objects:
        return None
    if isinstance(raw_objects, dict):
        return {str(label): _bbox_from_mapping(value) for label, value in raw_objects.items()}
    object_rois: dict[str, BoundingBox] = {}
    for item in raw_objects:
        label = str(item.get("label", f"object_{len(object_rois) + 1}"))
        object_rois[label] = _bbox_from_mapping(item.get("bbox", item))
    return object_rois


def _bbox_from_mapping(data: Any) -> BoundingBox:
    if not isinstance(data, dict):
        raise ValueError("Expected bbox object with x_min, y_min, x_max, y_max")
    return BoundingBox(
        x_min=float(data["x_min"]),
        y_min=float(data["y_min"]),
        x_max=float(data["x_max"]),
        y_max=float(data["y_max"]),
    )


def _dataset_info(dataset_path: Path, artifact_root: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    camera_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    total_episodes = int(info.get("total_episodes", 1))
    review_store = EpisodeReviewStore(artifact_root / "episode_reviews.json")
    episodes = [
        {
            "episode_index": episode_index,
            "review": review_store.get_review(dataset_path, episode_index),
        }
        for episode_index in range(total_episodes)
    ]
    return {
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": info.get("total_frames"),
        "camera_keys": camera_keys,
        "episodes": episodes,
    }


def _joint_comparison_from_query(query: dict[str, list[str]]) -> dict[str, Any]:
    reference_dataset_path = Path(_first_query_value(query, "reference_dataset_path", str(DEFAULT_FAILED_DATASET)))
    replay_dataset_path = Path(_first_query_value(query, "replay_dataset_path", str(DEFAULT_MOCK_ROBOT_DATASET)))
    reference_episode_index = int(_first_query_value(query, "reference_episode_index", "0"))
    reference_frame_index = int(_first_query_value(query, "reference_frame_index", "0"))
    replay_episode_index = int(_first_query_value(query, "replay_episode_index", "0"))
    replay_frame_index = int(_first_query_value(query, "replay_frame_index", "0"))
    urdf_path_value = _optional_str(_first_query_value(query, "urdf_path", ""))
    urdf_path = Path(urdf_path_value).expanduser() if urdf_path_value else None
    return compare_joint_positions(
        reference_dataset_path=reference_dataset_path,
        replay_dataset_path=replay_dataset_path,
        reference_episode_index=reference_episode_index,
        reference_frame_index=reference_frame_index,
        replay_episode_index=replay_episode_index,
        replay_frame_index=replay_frame_index,
        urdf_path=urdf_path,
    )


def compare_joint_positions(
    reference_dataset_path: Path,
    replay_dataset_path: Path,
    reference_episode_index: int,
    reference_frame_index: int,
    replay_episode_index: int,
    replay_frame_index: int,
    urdf_path: Path | None = None,
) -> dict[str, Any]:
    feature_names = _joint_feature_names(reference_dataset_path)
    urdf_joints = _read_urdf_joint_metadata(urdf_path) if urdf_path else {}
    reference_values = _read_joint_state(reference_dataset_path, reference_episode_index, reference_frame_index)
    replay_values = _read_joint_state(replay_dataset_path, replay_episode_index, replay_frame_index)
    joint_count = min(len(reference_values), len(replay_values), len(feature_names))
    joints = []
    for index in range(joint_count):
        feature_name = feature_names[index]
        joint_name = _normalize_joint_name(feature_name)
        urdf_metadata = urdf_joints.get(joint_name, {})
        reference_value = float(reference_values[index])
        replay_value = float(replay_values[index])
        joints.append(
            {
                "name": joint_name,
                "feature_name": feature_name,
                "reference": reference_value,
                "replay": replay_value,
                "delta": replay_value - reference_value,
                "lower": urdf_metadata.get("lower"),
                "upper": urdf_metadata.get("upper"),
                "type": urdf_metadata.get("type"),
                "in_urdf": joint_name in urdf_joints,
            }
        )
    return {
        "reference_dataset_path": str(reference_dataset_path),
        "replay_dataset_path": str(replay_dataset_path),
        "reference_episode_index": reference_episode_index,
        "reference_frame_index": reference_frame_index,
        "replay_episode_index": replay_episode_index,
        "replay_frame_index": replay_frame_index,
        "urdf_path": str(urdf_path) if urdf_path else None,
        "urdf_loaded": bool(urdf_joints),
        "joints": joints,
    }


def _joint_feature_names(dataset_path: Path) -> list[str]:
    info = _read_dataset_info(dataset_path)
    feature = info.get("features", {}).get("observation.state", {})
    names = feature.get("names")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    shape = feature.get("shape")
    joint_count = int(shape[0]) if isinstance(shape, list) and shape else 0
    return [f"joint_{index + 1}.pos" for index in range(joint_count)]


def _read_joint_state(dataset_path: Path, episode_index: int, frame_index: int) -> list[float]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Joint comparison requires pyarrow. Run the app with `uv run --extra dataset "
            "lerobot-pre-record-setup-match-web ...`."
        ) from exc

    for data_file in sorted(dataset_path.glob("data/**/*.parquet")):
        parquet_file = pq.ParquetFile(data_file)
        columns = set(parquet_file.schema_arrow.names)
        required_columns = {"episode_index", "frame_index", "observation.state"}
        if not required_columns.issubset(columns):
            continue
        table = pq.read_table(data_file, columns=sorted(required_columns))
        mask = pc.and_(
            pc.equal(table["episode_index"], episode_index),
            pc.equal(table["frame_index"], frame_index),
        )
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue
        values = filtered.column("observation.state")[0].as_py()
        return [float(value) for value in values]
    raise IndexError(
        f"No joint state found in {dataset_path} for episode={episode_index}, frame={frame_index}"
    )


def _read_urdf_joint_metadata(urdf_path: Path | None) -> dict[str, dict[str, Any]]:
    if urdf_path is None or not urdf_path.exists():
        return {}
    root = ElementTree.parse(urdf_path).getroot()
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type")
        if not name or joint_type == "fixed":
            continue
        limit = joint.find("limit")
        metadata: dict[str, Any] = {"type": joint_type}
        if limit is not None:
            if "lower" in limit.attrib:
                metadata["lower"] = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                metadata["upper"] = float(limit.attrib["upper"])
        joints[_normalize_joint_name(name)] = metadata
    return joints


def _normalize_joint_name(name: str) -> str:
    return name.removesuffix(".pos")


def _read_dataset_info(dataset_path: Path) -> dict[str, Any]:
    with (dataset_path / "meta" / "info.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_episode_frame(dataset_path: Path, camera_key: str, episode_index: int, frame_index: int) -> np.ndarray:
    source = EpisodeReplayFrameSource(dataset_path=dataset_path, camera_key=camera_key, episode_index=episode_index)
    return source.read_frame(frame_index)


def _frame_payload(frame_bgr: np.ndarray) -> dict[str, Any]:
    return {
        "width": int(frame_bgr.shape[1]),
        "height": int(frame_bgr.shape[0]),
        "image": _encode_jpeg_data_url(frame_bgr),
    }


def _encode_jpeg_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        raise OSError("Could not encode frame as JPEG")
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


def _stream_match_payloads(config: dict[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    scenario = _scenario_from_payload(config)
    object_rois = scenario.object_rois or {"reference_crop": BoundingBox(0.35, 0.35, 0.65, 0.65)}
    detector = TemplateMatchingDetector()
    reference_source = EpisodeReplayFrameSource(
        dataset_path=scenario.reference_dataset_path,
        camera_key=scenario.camera_key,
        episode_index=scenario.reference_episode_index,
    )
    replay_source = EpisodeReplayFrameSource(
        dataset_path=scenario.replay_dataset_path,
        camera_key=scenario.camera_key,
        episode_index=scenario.replay_episode_index,
    )
    reference_frame = reference_source.read_frame(scenario.reference_frame_index)
    profile = SetupProfile(
        profile_id=scenario.profile_id,
        objects=[
            create_object_template(reference_frame, label, scenario.camera_key, bbox)
            for label, bbox in object_rois.items()
        ],
    )
    reference_state = ReferenceState(
        reference_id=f"reference-{uuid4().hex[:12]}",
        profile_id=profile.profile_id,
        dataset_path=str(scenario.reference_dataset_path),
        episode_index=scenario.reference_episode_index,
        frame_index=scenario.reference_frame_index,
        camera_key=scenario.camera_key,
        target_centroids={label: bbox.centroid for label, bbox in object_rois.items()},
    )
    stream_frames = max(1, scenario.stream_frames)
    results: list[dict[str, Any]] = []
    final_match_result: MatchResult | None = None
    frame_results: list[dict[str, Any]] = []
    for stream_frame_index in range(stream_frames):
        replay_frame_index = scenario.replay_start_frame_index + stream_frame_index
        replay_frame = replay_source.read_frame(replay_frame_index)
        detections = detector.detect(replay_frame, profile, scenario.camera_key)
        ready, overall_score, object_scores, warnings = score_setup(profile, reference_state, detections, ScoreConfig())
        warnings = [*warnings, *_camera_drift_warnings(reference_frame, replay_frame)]
        frame_result = {
            "frame_index": replay_frame_index,
            "ready": ready,
            "overall_score": overall_score,
            "detections": [detection.to_dict() for detection in detections],
            "object_scores": [object_score.to_dict() for object_score in object_scores],
            "warnings": warnings,
        }
        frame_results.append(frame_result)
        final_match_result = MatchResult(
            match_id=f"match-{uuid4().hex[:12]}",
            profile_id=profile.profile_id,
            reference_id=reference_state.reference_id,
            reference_dataset_path=str(scenario.reference_dataset_path),
            reference_episode_index=scenario.reference_episode_index,
            reference_frame_index=scenario.reference_frame_index,
            replay_dataset_path=str(scenario.replay_dataset_path),
            replay_episode_index=scenario.replay_episode_index,
            camera_key=scenario.camera_key,
            ready=ready,
            overall_score=overall_score,
            object_scores=object_scores,
            warnings=warnings,
            frame_results=list(frame_results),
        )
        results.append(
            {
                "type": "frame",
                "frame": _frame_payload(replay_frame),
                "reference_frame": _frame_payload(reference_frame) if stream_frame_index == 0 else None,
                **frame_result,
            }
        )
    if bool(config.get("save", True)) and final_match_result is not None:
        artifact_store = ArtifactStore(artifact_root)
        artifact_store.save_profile(profile)
        artifact_store.save_reference_state(reference_state)
        artifact_store.save_match_result(final_match_result)
    if final_match_result is not None:
        results.append({"type": "result", "match_result": final_match_result.to_dict(), "artifact_root": str(artifact_root)})
    return results


def _refine_bbox_with_color_blob(frame_bgr: np.ndarray, bbox: BoundingBox) -> BoundingBox:
    rows, cols = bbox.to_pixel_slice(*frame_bgr.shape[:2])
    crop = frame_bgr[rows, cols]
    if crop.size == 0:
        return bbox
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 45) & (value > 35)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 12:
        return bbox
    x, y, width, height = cv2.boundingRect(contour)
    frame_height, frame_width = frame_bgr.shape[:2]
    col_offset = cols.start or 0
    row_offset = rows.start or 0
    padding = 3
    x_min = max(0, col_offset + x - padding) / frame_width
    y_min = max(0, row_offset + y - padding) / frame_height
    x_max = min(frame_width, col_offset + x + width + padding) / frame_width
    y_max = min(frame_height, row_offset + y + height + padding) / frame_height
    if math.isclose(x_min, x_max) or math.isclose(y_min, y_max):
        return bbox
    return BoundingBox(x_min, y_min, x_max, y_max)


def _websocket_read_text(stream: Any) -> str:
    header = stream.read(2)
    if len(header) != 2:
        raise ConnectionError("WebSocket client disconnected before sending configuration")
    first_byte, second_byte = header
    opcode = first_byte & 0x0F
    if opcode == 8:
        raise ConnectionError("WebSocket closed")
    masked = bool(second_byte & 0x80)
    length = second_byte & 0x7F
    if length == 126:
        length = struct.unpack("!H", stream.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", stream.read(8))[0]
    mask = stream.read(4) if masked else b""
    payload = bytearray(stream.read(length))
    if masked:
        for index in range(length):
            payload[index] ^= mask[index % 4]
    return payload.decode("utf-8")


def _websocket_write_text(stream: Any, message: str) -> None:
    payload = message.encode("utf-8")
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x81, length)
    elif length < 65536:
        header = struct.pack("!BBH", 0x81, 126, length)
    else:
        header = struct.pack("!BBQ", 0x81, 127, length)
    stream.write(header + payload)
    stream.flush()


def _first_query_value(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
