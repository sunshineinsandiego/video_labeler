"""
CoMotion MOT-style tracks (.txt) parsing and extended rows (MOT CSV + tab suffix).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# CoMotion: MOT column 0 is 1-based (frame 1 = first video frame); UI / OpenCV use 0-based frames.
MOT_FRAME_OFFSET = 1

# CoMotion `convert_to_mot` emits exactly 10 comma-separated fields per row.
MOT_FIELD_COUNT = 10


def ui_frame_from_mot(mot_frame: int) -> int:
    return int(mot_frame) - MOT_FRAME_OFFSET


def mot_frame_from_ui(frame_ui: int) -> int:
    return int(frame_ui) + MOT_FRAME_OFFSET


def split_extended_mot_line(line: str) -> Tuple[str, str, str, str]:
    """Split '<mot_csv>\\tname\\taction\\tmanual' into parts."""
    line = line.rstrip("\n\r")
    if "\t" not in line:
        return line, "", "", ""
    mot_csv, rest = line.split("\t", 1)
    parts = rest.split("\t", 2)
    name = parts[0] if len(parts) > 0 else ""
    action = parts[1] if len(parts) > 1 else ""
    manual = parts[2] if len(parts) > 2 else ""
    return mot_csv, name, action, manual


def format_extended_mot_line(mot_csv: str, name: str, action: str, manual: str) -> str:
    return f"{mot_csv}\t{name}\t{action}\t{manual}"


def parse_mot_bbox_only(mot_csv: str) -> Optional[Tuple[int, int, List[float]]]:
    """
    Parse MOT columns: frame, id, L, T, W, H, conf, x, y, z (exactly 10 fields).
    Undo CoMotion +1 on L,T,W,H.
    Returns (mot_frame, track_id, bbox x1y1x2y2).
    Returns None only when mot_csv is empty/whitespace.
    Raises ValueError when mot_csv is non-empty but not a valid 10-field CoMotion row.
    """
    mot_csv = mot_csv.strip()
    if not mot_csv:
        return None
    try:
        row = next(csv.reader([mot_csv]))
    except Exception as e:
        raise ValueError(f"Invalid MOT CSV: {mot_csv!r}") from e
    if len(row) != MOT_FIELD_COUNT:
        raise ValueError(
            f"MOT row must have exactly {MOT_FIELD_COUNT} comma-separated fields (CoMotion export); "
            f"got {len(row)} in {mot_csv!r}"
        )
    try:
        mot_frame = int(float(row[0]))
        track_id = int(float(row[1]))
        l, t, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except ValueError as e:
        raise ValueError(f"Non-numeric MOT frame/id/bbox in {mot_csv!r}") from e
    l0, t0, w0, h0 = l - 1.0, t - 1.0, w - 1.0, h - 1.0
    x1, y1, x2, y2 = l0, t0, l0 + w0, t0 + h0
    return mot_frame, track_id, [x1, y1, x2, y2]


def load_mot_tracks_from_text(text: str) -> Dict[int, List[Dict[str, Any]]]:
    """tracks_by_frame keyed by 0-based UI frame (hard-fail on malformed rows)."""
    tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    det_ids: Dict[int, int] = {}
    for line_num, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        mot_csv, _, _, _ = split_extended_mot_line(line)
        try:
            parsed = parse_mot_bbox_only(mot_csv)
        except ValueError as e:
            raise ValueError(f"Line {line_num}: {e}") from e
        if not parsed:
            raise ValueError(f"Malformed MOT row at line {line_num}: empty MOT prefix in {line[:120]!r}")
        mot_f, tid, bbox = parsed
        frame_ui = ui_frame_from_mot(mot_f)
        det = det_ids.get(frame_ui, 0)
        det_ids[frame_ui] = det + 1
        tracks_by_frame.setdefault(frame_ui, []).append(
            {
                "frame": frame_ui,
                "mot_frame": mot_f,
                "track_id": tid,
                "det_id": det,
                "bbox": bbox,
                "score": 1.0,
            }
        )
    return tracks_by_frame


def load_mot_tracks_from_path(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    return load_mot_tracks_from_text(path.read_text(encoding="utf-8"))


def ensure_extended_mot_copy(tracks_src: Path, dest: Path) -> None:
    """Create extended MOT file from raw tracks: append empty name/action/manual columns."""
    if dest.exists():
        return
    if not tracks_src.exists():
        dest.write_text("", encoding="utf-8")
        return
    out_lines = []
    for line_num, line in enumerate(tracks_src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        mot_csv, _, _, _ = split_extended_mot_line(line)
        mot_csv = mot_csv.strip()
        try:
            parsed = parse_mot_bbox_only(mot_csv)
        except ValueError as e:
            raise ValueError(f"tracks source line {line_num}: {e}") from e
        if not parsed:
            raise ValueError(f"tracks source line {line_num}: empty MOT prefix")
        out_lines.append(format_extended_mot_line(mot_csv, "", "", ""))
    dest.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def _manual_json_from_payload(payload: Dict[str, Any]) -> str:
    manual = {
        "keypoints": payload.get("keypoints") or [],
        "lines": payload.get("lines") or [],
        "rois": payload.get("rois") or [],
        "measurements": payload.get("measurements") or {"distances": [], "angles": []},
    }
    return json.dumps(manual, ensure_ascii=False)


def _normalize_manual_annotations(raw: Any) -> str:
    if not isinstance(raw, dict):
        raw = {}
    manual = {
        "keypoints": raw.get("keypoints") or [],
        "lines": raw.get("lines") or [],
        "rois": raw.get("rois") or [],
        "measurements": raw.get("measurements") or {"distances": [], "angles": []},
    }
    return json.dumps(manual, ensure_ascii=False)


def frame_payload_from_extended_lines(lines: List[str], frame_ui: int) -> Dict[str, Any]:
    """Build annotation dict for one UI frame from extended MOT lines."""
    target_mot = mot_frame_from_ui(frame_ui)
    bounding_boxes: Dict[str, Any] = {}
    merged_manual: Optional[Dict[str, Any]] = None
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        mot_csv, name, action, manual = split_extended_mot_line(line)
        try:
            parsed = parse_mot_bbox_only(mot_csv)
        except ValueError as e:
            raise ValueError(f"Extended MOT line {line_num}: {e}") from e
        if not parsed:
            raise ValueError(f"Extended MOT line {line_num}: empty MOT prefix")
        mot_f, tid, _bbox = parsed
        if mot_f != target_mot:
            continue
        sk = str(tid)
        bounding_boxes[sk] = {
            "track_id": tid,
            "name": name or None,
            "action": action or None,
        }
        if manual and manual.strip():
            try:
                parsed_manual = json.loads(manual)
                if isinstance(parsed_manual, dict):
                    bounding_boxes[sk]["annotations"] = parsed_manual
                    if merged_manual is None:
                        merged_manual = parsed_manual
            except json.JSONDecodeError:
                pass
    keypoints: List[Any] = []
    lines_d: List[Any] = []
    rois: List[Any] = []
    distances: List[Any] = []
    angles: List[Any] = []
    if isinstance(merged_manual, dict):
        keypoints = merged_manual.get("keypoints") or []
        lines_d = merged_manual.get("lines") or []
        rois = merged_manual.get("rois") or []
        measurements = merged_manual.get("measurements") or {}
        distances = measurements.get("distances") or []
        angles = measurements.get("angles") or []
    return {
        "bounding_boxes": bounding_boxes,
        "keypoints": keypoints,
        "lines": lines_d,
        "rois": rois,
        "measurements": {"distances": distances, "angles": angles},
    }


def sync_payload_to_extended_mot_file(path: Path, frame_ui: int, payload: Dict[str, Any]) -> None:
    """Update name/action/manual on MOT rows for this UI frame (manual is per track row)."""
    if not path.exists():
        return
    target_mot = mot_frame_from_ui(frame_ui)
    lines = path.read_text(encoding="utf-8").splitlines()
    bb = payload.get("bounding_boxes") or {}
    selected_track_id = payload.get("selected_track_id")
    global_manual_json = _manual_json_from_payload(payload)
    new_lines: List[str] = []
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            new_lines.append(line)
            continue
        mot_csv, old_name, old_action, _old_manual = split_extended_mot_line(line)
        try:
            parsed = parse_mot_bbox_only(mot_csv)
        except ValueError as e:
            raise ValueError(f"Extended MOT line {line_num}: {e}") from e
        if not parsed:
            raise ValueError(f"Extended MOT line {line_num}: empty MOT prefix")
        mot_f, tid, __ = parsed
        if mot_f != target_mot:
            new_lines.append(line)
            continue
        sk = str(tid)
        name = old_name
        action = old_action
        manual_json = _old_manual
        if sk in bb:
            bd = bb[sk]
            if isinstance(bd, dict):
                if "name" in bd and bd["name"] is not None:
                    name = str(bd["name"])
                if "action" in bd and bd["action"] is not None:
                    action = str(bd["action"])
                if "annotations" in bd:
                    manual_json = _normalize_manual_annotations(bd.get("annotations"))
                elif selected_track_id is not None and str(selected_track_id) == sk:
                    manual_json = global_manual_json
        new_lines.append(format_extended_mot_line(mot_csv.strip(), name, action, manual_json))
    path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
