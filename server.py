import os
import json
import logging
import secrets
import shutil
import sqlite3
import traceback
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from threading import Lock

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from passlib.hash import bcrypt

from mot_tracks import (
    ensure_extended_mot_copy,
    frame_payload_from_extended_lines,
    load_mot_tracks_from_path,
    load_mot_tracks_from_text,
    sync_payload_to_extended_mot_file,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
STUDY_BASE = DATA_DIR / "studies"
TEMP_BASE = DATA_DIR / "temp"
USERS_DB = DATA_DIR / "users.db"

ADMIN_EMAIL = "cd2859@cumc.columbia.edu"
SESSION_COOKIE_NAME = "session_id"
RATE_LIMIT_WINDOW = 300
RATE_LIMIT_MAX = 5

# Create directories
for path in (STATIC_DIR, STUDY_BASE, TEMP_BASE):
    os.makedirs(path, exist_ok=True)

# Clear temp directory on startup
if TEMP_BASE.exists():
    try:
        for item in TEMP_BASE.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        logger.info(f"Cleared temp directory: {TEMP_BASE}")
    except Exception as e:
        logger.warning(f"Error clearing temp directory: {e}")

SESSIONS: Dict[str, Dict[str, Any]] = {}
RATE_LIMITS: Dict[str, List[float]] = {}


def init_user_db() -> None:
    with sqlite3.connect(USERS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_user_db()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _sanitize_segment(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip().lower())
    return cleaned or fallback


def _safe_study_id(study_id: str) -> str:
    cleaned = study_id.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Study ID is required")
    parts = Path(cleaned).parts
    if len(parts) != 1 or parts[0] in (".", "..") or "\\" in cleaned:
        raise HTTPException(status_code=400, detail="Invalid study ID")
    return cleaned


def _safe_filename(filename: str) -> str:
    if not filename:
        return ""
    normalized = filename.replace("\\", "/")
    base = Path(normalized).name
    if base in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base


def _user_key(email: str) -> str:
    local = email.split("@", 1)[0].strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in local)
    return safe or "user"


def _ensure_user_dirs(user_key: str) -> None:
    (STUDY_BASE / user_key).mkdir(parents=True, exist_ok=True)
    (TEMP_BASE / user_key).mkdir(parents=True, exist_ok=True)


def _user_dirs(request: Request) -> tuple[str, Path, Path]:
    user = getattr(request.state, "user", None)
    if not user or "user_key" not in user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    key = user["user_key"]
    _ensure_user_dirs(key)
    return key, STUDY_BASE / key, TEMP_BASE / key


def _get_user(email: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(USERS_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT email, password_hash, is_admin FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def _rate_key(request: Request, email: str) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{ip}:{email}"


def _prune_attempts(attempts: List[float], now: float) -> List[float]:
    cutoff = now - RATE_LIMIT_WINDOW
    return [t for t in attempts if t >= cutoff]


def _is_rate_limited(key: str) -> bool:
    now = time.time()
    attempts = _prune_attempts(RATE_LIMITS.get(key, []), now)
    RATE_LIMITS[key] = attempts
    return len(attempts) >= RATE_LIMIT_MAX


def _record_attempt(key: str) -> None:
    now = time.time()
    attempts = _prune_attempts(RATE_LIMITS.get(key, []), now)
    attempts.append(now)
    RATE_LIMITS[key] = attempts


def _clear_attempts(key: str) -> None:
    RATE_LIMITS.pop(key, None)


def _create_session(email: str, is_admin: bool) -> str:
    session_id = secrets.token_urlsafe(32)
    user_key = _user_key(email)
    _ensure_user_dirs(user_key)
    SESSIONS[session_id] = {
        "email": email,
        "user_key": user_key,
        "is_admin": is_admin,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return session_id


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def _render_login_page(error: Optional[str] = None) -> str:
    error_block = ""
    if error:
        error_block = f"<div class=\"error\">{error}</div>"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Video Annotate</title>
    <style>
      body {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        background: #0f1117;
        color: #f5f7ff;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        margin: 0;
      }}
      .card {{
        background: #1b1f2a;
        padding: 2rem;
        border-radius: 12px;
        width: 100%;
        max-width: 380px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      }}
      h1 {{
        margin: 0 0 1.5rem 0;
        font-size: 1.6rem;
        font-weight: 700;
      }}
      label {{
        display: block;
        margin: 0 0 0.4rem 0;
        font-size: 0.9rem;
      }}
      input {{
        width: 100%;
        padding: 0.6rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 6px;
        border: 1px solid #2f3546;
        background: #10131c;
        color: #f5f7ff;
        box-sizing: border-box;
      }}
      button {{
        width: 100%;
        padding: 0.65rem 0.8rem;
        border-radius: 6px;
        border: none;
        background: #4fc3f7;
        font-weight: 600;
        cursor: pointer;
      }}
      .error {{
        background: #2c1b1f;
        border: 1px solid #7f1d1d;
        color: #fecaca;
        padding: 0.6rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Video Annotate</h1>
      {error_block}
      <form method="post" action="/auth/login">
        <label for="email">Email</label>
        <input id="email" name="email" type="email" required autofocus />
        <label for="password">Password</label>
        <input id="password" name="password" type="password" required />
        <button type="submit">Login</button>
      </form>
    </div>
  </body>
</html>"""

app = FastAPI(title="Video Labeler")

# Allow local development UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {"/login", "/auth/login", "/favicon.ico"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS:
        return await call_next(request)
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id and session_id in SESSIONS:
        request.state.user = SESSIONS[session_id]
        return await call_next(request)
    if _wants_html(request):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/cache/clear")
async def clear_video_cache(request: Request) -> JSONResponse:
    """Manually clear the video capture cache for the current user."""
    user_key, _, _ = _user_dirs(request)
    with _video_cache_lock:
        user_cache = _user_video_cache(user_key)
        count = len(user_cache)
        for _, cache_entry in list(user_cache.items()):
            try:
                cache_entry["capture"].release()
            except Exception:
                pass
        user_cache.clear()
    
    logger.info(f"Cleared video cache for {user_key}: {count} video captures released")
    return JSONResponse({"status": "cleared", "count": count})


@app.get("/favicon.ico")
async def favicon():
    """Suppress favicon 404 errors."""
    raise HTTPException(status_code=404)


@app.get("/login")
async def login_page(error: Optional[str] = None) -> HTMLResponse:
    return HTMLResponse(_render_login_page(error))


@app.post("/auth/login")
async def login_action(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    email_norm = _normalize_email(email)
    if not email_norm or not password:
        return HTMLResponse(_render_login_page("Email and password are required."), status_code=400)

    key = _rate_key(request, email_norm)
    if _is_rate_limited(key):
        return HTMLResponse(_render_login_page("Too many attempts. Try again later."), status_code=429)

    user = _get_user(email_norm)
    if not user or not bcrypt.verify(password, user["password_hash"]):
        _record_attempt(key)
        return HTMLResponse(_render_login_page("Invalid email or password."), status_code=401)

    _clear_attempts(key)
    session_id = _create_session(user["email"], bool(user["is_admin"]))
    response = RedirectResponse("/", status_code=303)
    secure_cookie = request.url.scheme == "https"
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="strict",
        secure=secure_cookie,
    )
    return response


@app.get("/README.html")
async def help_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "README.html")

@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


def get_video_capture(user_key: str, video_id: str, video_path: Path) -> Dict[str, Any]:
    """
    Get a cached video capture object, or create a new one.
    Keeps video files open in memory for faster frame access.
    """
    current_time = time.time()
    
    with _video_cache_lock:
        user_cache = _user_video_cache(user_key)
        # Check if we have a cached capture for this video
        if video_id in user_cache:
            cache_entry = user_cache[video_id]
            
            # Check if capture is still valid
            if cache_entry["capture"].isOpened():
                # Update last access time
                cache_entry["last_access"] = current_time
                cache_entry.setdefault("lock", Lock())
                logger.debug(f"Reusing cached video capture for {video_id}")
                return cache_entry
            else:
                # Capture is closed, remove from cache
                logger.info(f"Cached video capture for {video_id} was closed, removing from cache")
                del user_cache[video_id]
        
        # Clean up stale cache entries (older than TTL)
        stale_keys = []
        for vid, cache_entry in user_cache.items():
            if current_time - cache_entry["last_access"] > VIDEO_CACHE_TTL:
                stale_keys.append(vid)
        
        for vid in stale_keys:
            logger.info(f"Closing stale video capture for {vid} (inactive for {VIDEO_CACHE_TTL}s)")
            try:
                user_cache[vid]["capture"].release()
            except Exception:
                pass
            del user_cache[vid]
        
        # Create new capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Cache it
        user_cache[video_id] = {
            "capture": cap,
            "path": video_path,
            "last_access": current_time,
            "lock": Lock(),
        }
        
        logger.info(f"Created new cached video capture for {video_id}")
        return user_cache[video_id]


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video metadata without extracting frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _temp_tracks_path(temp_root: Path, video_id: str) -> Path:
    return temp_root / f"tracks_{video_id}.txt"


def _temp_annotations_txt(temp_root: Path, study_id: str) -> Path:
    return temp_root / f"{study_id}_annotations.txt"


def _study_saved_tracks_path(study_dir: Path, metadata: Dict[str, Any]) -> Optional[Path]:
    name = metadata.get("tracks_txt_filename")
    if name:
        try:
            safe = _safe_filename(str(name))
        except HTTPException:
            safe = ""
        if safe:
            p = study_dir / safe
            if p.exists():
                return p
    for f in sorted(study_dir.iterdir()):
        if f.suffix.lower() == ".txt" and "track" in f.name.lower() and not f.name.endswith("_annotations.txt"):
            return f
    for f in sorted(study_dir.iterdir()):
        if f.suffix.lower() == ".txt" and not f.name.endswith("_annotations.txt"):
            return f
    return None


def _merge_frame_payload_disk_and_memory(disk: Dict[str, Any], mem: Dict[str, Any]) -> Dict[str, Any]:
    """Merge on-disk extended-MOT frame payload with in-memory edits before sync."""
    out: Dict[str, Any] = {
        "bounding_boxes": dict(disk.get("bounding_boxes") or {}),
        "keypoints": list(disk.get("keypoints") or []),
        "lines": list(disk.get("lines") or []),
        "rois": list(disk.get("rois") or []),
        "measurements": dict(disk.get("measurements") or {"distances": [], "angles": []}),
    }
    if "keypoints" in mem:
        kp = mem["keypoints"]
        out["keypoints"] = [] if kp is None else list(kp)
    if "lines" in mem:
        ln = mem["lines"]
        out["lines"] = [] if ln is None else list(ln)
    if "rois" in mem:
        r = mem["rois"]
        out["rois"] = [] if r is None else list(r)
    if "measurements" in mem and isinstance(mem.get("measurements"), dict):
        out["measurements"] = dict(mem["measurements"])
    mem_bb = mem.get("bounding_boxes") or {}
    for tid, bb in mem_bb.items():
        if not isinstance(bb, dict):
            continue
        sk = str(tid)
        if sk not in out["bounding_boxes"]:
            out["bounding_boxes"][sk] = dict(bb)
        else:
            merged = dict(out["bounding_boxes"][sk])
            merged.update(bb)
            out["bounding_boxes"][sk] = merged
    return out


def _flush_study_memory_to_extended_mot(
    user_key: str,
    temp_root: Path,
    study_id: str,
    video_id: str,
) -> None:
    """Write all in-memory frame annotations for this study into the extended MOT file."""
    fa = _user_frame_annotations(user_key).get(study_id, {})
    if not fa:
        return
    ann_path = _temp_annotations_txt(temp_root, study_id)
    tracks_path = _temp_tracks_path(temp_root, video_id)
    ensure_extended_mot_copy(tracks_path, ann_path)
    lines = ann_path.read_text(encoding="utf-8").splitlines() if ann_path.exists() else []
    for frame_idx in sorted(fa.keys(), key=lambda x: int(x) if isinstance(x, str) and str(x).isdigit() else x):
        mem = fa.get(frame_idx)
        if not isinstance(mem, dict):
            continue
        try:
            fi = int(frame_idx) if isinstance(frame_idx, str) and str(frame_idx).isdigit() else int(frame_idx)
        except Exception:
            continue
        disk = frame_payload_from_extended_lines(lines, fi)
        merged = _merge_frame_payload_disk_and_memory(disk, mem)
        sync_payload_to_extended_mot_file(ann_path, fi, merged)
        lines = ann_path.read_text(encoding="utf-8").splitlines()


@app.post("/upload")
async def upload_file(
    request: Request,
    video_file: UploadFile = File(...),
    tracks_txt_file: UploadFile = File(...),
) -> JSONResponse:
    logger.info(
        f"Upload: video={video_file.filename}, tracks_txt={tracks_txt_file.filename if tracks_txt_file else None}"
    )
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Video filename is required.")
    if tracks_txt_file is None or not tracks_txt_file.filename:
        raise HTTPException(status_code=400, detail="Tracks .txt file is required (MOT format).")
    if Path(_safe_filename(tracks_txt_file.filename)).suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="Tracks file must be a .txt file.")

    user_key, study_root, temp_root = _user_dirs(request)
    _clear_user_temp(user_key, temp_root)

    video_extension = Path(video_file.filename).suffix.lower()
    if video_extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    video_id = uuid.uuid4().hex
    video_path = temp_root / f"video_{video_id}{video_extension}"

    try:
        with video_path.open("wb") as buffer:
            while True:
                chunk = await video_file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store video: {exc}") from exc

    tracks_original_name = _safe_filename(tracks_txt_file.filename)
    tracks_path = _temp_tracks_path(temp_root, video_id)
    try:
        with tracks_path.open("wb") as buffer:
            while True:
                chunk = await tracks_txt_file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except OSError as exc:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to store tracks: {exc}") from exc

    try:
        video_info = get_video_info(video_path)
        total_frames = video_info["total_frames"]
        if total_frames == 0:
            raise HTTPException(status_code=400, detail="Video has no frames.")
        frames = [
            {
                "frame_index": i,
                "timestamp": i / video_info["fps"] if video_info["fps"] > 0 else 0.0,
            }
            for i in range(total_frames)
        ]
    except HTTPException:
        raise
    except Exception as exc:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        if tracks_path.exists():
            tracks_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to read video: {exc}") from exc

    try:
        keypoints_tracks = load_mot_tracks_from_path(tracks_path)
    except Exception as exc:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        if tracks_path.exists():
            tracks_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid MOT tracks file: {exc}") from exc

    first_image_url = f"/video/{video_id}/frame/0"
    meta = {
        "video_filename": _safe_filename(video_file.filename),
        "tracks_txt_filename": tracks_original_name,
        "total_frames": len(frames),
        "has_keypoints": len(keypoints_tracks) > 0,
        "tracks_source_format": "mot_txt",
    }
    response_data = {
        "video_id": video_id,
        "image_url": first_image_url,
        "stored_filename": video_path.name,
        "kind": "video",
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": len(frames),
        "metadata": meta,
    }
    _user_temp_studies(user_key)[video_id] = {
        "video_id": video_id,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": len(frames),
        "metadata": meta,
    }
    return JSONResponse(response_data)


@app.get("/video/{video_id}/frame/{frame_index}")
async def get_frame(request: Request, video_id: str, frame_index: int) -> Response:
    """Get a specific frame image by streaming from video file (with caching)."""
    user_key, _, temp_root = _user_dirs(request)
    # Find the video file
    video_path = None
    
    # Try common video extensions
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        potential_path = temp_root / f"video_{video_id}{ext}"
        if potential_path.exists():
            video_path = potential_path
            break
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found for video_id: {video_id}")
    
    try:
        # Get cached video capture (or create new one)
        # This keeps the video file open in SERVER memory, not browser
        cache_entry = get_video_capture(user_key, video_id, video_path)
        cap = cache_entry["capture"]
        lock = cache_entry.get("lock")
        
        # Seek to the requested frame
        if lock:
            with lock:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
        
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_index} not found in video")
        
        # Encode frame as JPEG (more efficient than PNG for streaming)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = encoded_img.tobytes()
        
        # Return frame as JPEG (browser only receives this single frame, not the whole video)
        return Response(content=frame_bytes, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting frame {frame_index} from video {video_id}: {e}")
        # On error, try to remove from cache and retry once
        with _video_cache_lock:
            user_cache = _user_video_cache(user_key)
            if video_id in user_cache:
                try:
                    user_cache[video_id]["capture"].release()
                except Exception:
                    pass
                del user_cache[video_id]
        raise HTTPException(status_code=500, detail=f"Failed to extract frame: {str(e)}")


@app.get("/video/{video_id}/keypoints/{frame_index}")
async def get_frame_keypoints(request: Request, video_id: str, frame_index: int) -> JSONResponse:
    """Get tracks (bboxes) for a specific frame from cached MOT data or disk."""
    user_key, _, temp_root = _user_dirs(request)
    temp_data = _user_temp_studies(user_key).get(video_id)
    if temp_data and temp_data.get("keypoints_tracks"):
        keypoints_tracks = temp_data["keypoints_tracks"]
    else:
        tp = _temp_tracks_path(temp_root, video_id)
        if tp.exists():
            keypoints_tracks = load_mot_tracks_from_path(tp)
        else:
            keypoints_tracks = {}
    frame_tracks = keypoints_tracks.get(frame_index, [])
    if not frame_tracks:
        frame_tracks = keypoints_tracks.get(str(frame_index), [])
    return JSONResponse({"tracks": frame_tracks})


# Store annotations per frame in memory, per user
_frame_annotations: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

# Store video metadata (keypoints_tracks, frames, etc.) for temp studies, per user
_temp_studies: Dict[str, Dict[str, Dict[str, Any]]] = {}

# Video capture cache to avoid reopening video files repeatedly (per user)
# Keeps video files open in SERVER memory for faster frame access
_video_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {user_key: {video_id: {"capture": cv2.VideoCapture, "path": Path, "last_access": time.time()}}}
_video_cache_lock = Lock()
_temp_annotation_locks: Dict[str, Lock] = {}


def _user_frame_annotations(user_key: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    return _frame_annotations.setdefault(user_key, {})


def _user_temp_studies(user_key: str) -> Dict[str, Dict[str, Any]]:
    return _temp_studies.setdefault(user_key, {})


def _user_video_cache(user_key: str) -> Dict[str, Dict[str, Any]]:
    return _video_cache.setdefault(user_key, {})


def _temp_lock(user_key: str, study_id: str) -> Lock:
    key = f"{user_key}:{study_id}"
    lock = _temp_annotation_locks.get(key)
    if lock is None:
        lock = Lock()
        _temp_annotation_locks[key] = lock
    return lock


def _clear_user_temp(user_key: str, temp_root: Path) -> None:
    # Clear in-memory annotations and temp studies for this user
    _user_frame_annotations(user_key).clear()
    _user_temp_studies(user_key).clear()

    # Clear video cache entries for this user
    with _video_cache_lock:
        user_cache = _user_video_cache(user_key)
        for _, cache_entry in list(user_cache.items()):
            try:
                cache_entry["capture"].release()
            except Exception:
                pass
        user_cache.clear()

    # Clear temp folder files
    if temp_root.exists():
        for item in temp_root.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Failed to delete temp item {item}: {e}")
VIDEO_CACHE_TTL = 300  # 5 minutes in seconds


@app.post("/study/{study_id}/frame/{frame_index}/annotations")
async def save_frame_annotations(
    request: Request,
    study_id: str,
    frame_index: int,
    payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Save annotations for a specific frame (updates temp extended MOT file)."""
    study_id = _safe_study_id(study_id)
    user_key, study_root, temp_root = _user_dirs(request)
    frame_annotations = _user_frame_annotations(user_key)
    if study_id not in frame_annotations:
        frame_annotations[study_id] = {}

    payload["frame"] = frame_index
    video_id = payload.get("video_id")
    bounding_boxes = payload.get("bounding_boxes")
    if isinstance(bounding_boxes, dict):
        for track_id_key, bbox_ann in bounding_boxes.items():
            if not isinstance(bbox_ann, dict):
                continue
            try:
                track_id = int(track_id_key) if isinstance(track_id_key, str) and track_id_key.isdigit() else track_id_key
            except Exception:
                track_id = track_id_key
            bbox_ann.setdefault("track_id", track_id)
            bbox_ann.setdefault("det_id", None)

        if video_id:
            study_data = _study_data_for_propagation(user_key, study_root, temp_root, study_id, video_id)
            keypoints_tracks = study_data.get("keypoints_tracks", {}) if study_data else {}
            if keypoints_tracks:
                for track_id_key, bbox_ann in bounding_boxes.items():
                    if not isinstance(bbox_ann, dict):
                        continue
                    if bbox_ann.get("det_id") is None:
                        match = _find_track_for_frame(
                            keypoints_tracks,
                            frame_index,
                            bbox_ann.get("track_id", track_id_key),
                        )
                        if match:
                            bbox_ann["track_id"] = match.get("track_id", bbox_ann.get("track_id"))
                            bbox_ann["det_id"] = match.get("det_id")

    frame_annotations[study_id][frame_index] = payload

    if video_id:
        ann_path = _temp_annotations_txt(temp_root, study_id)
        ensure_extended_mot_copy(_temp_tracks_path(temp_root, video_id), ann_path)
        try:
            with _temp_lock(user_key, study_id):
                sync_payload_to_extended_mot_file(ann_path, frame_index, payload)
        except ValueError as e:
            logger.error(f"Extended MOT sync rejected: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.warning(f"Failed to sync annotations txt: {e}")
            logger.error(traceback.format_exc())

    kp = len(payload.get("keypoints", []))
    ln = len(payload.get("lines", []))
    logger.info(f"Saved frame {frame_index} study {study_id}: keypoints={kp}, lines={ln}")
    return JSONResponse({"status": "saved"})


@app.get("/study/{study_id}/frame/{frame_index}/annotations")
async def get_frame_annotations(
    request: Request,
    study_id: str,
    frame_index: int,
    video_id: str = Query(..., description="Current session temp video id (required when reading from disk)."),
) -> JSONResponse:
    """Get annotations for a frame from memory or temp extended MOT (strict: temp ann + temp tracks only)."""
    study_id = _safe_study_id(study_id)
    user_key, _, temp_root = _user_dirs(request)
    frame_store = _user_frame_annotations(user_key)
    per_study = frame_store.get(study_id, {})
    annotations = per_study.get(frame_index) or per_study.get(str(frame_index))
    if annotations:
        return JSONResponse(annotations)

    vid = str(video_id).strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id query parameter is required and non-empty.")

    tracks_path = _temp_tracks_path(temp_root, vid)
    if not tracks_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No temp MOT tracks file for video_id {vid}; upload again or load a study.",
        )

    ann_temp = _temp_annotations_txt(temp_root, study_id)
    if not ann_temp.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Extended annotations file missing in temp for study {study_id}; upload/load study or POST a frame first.",
        )

    try:
        lines = ann_temp.read_text(encoding="utf-8").splitlines()
        man = frame_payload_from_extended_lines(lines, frame_index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid extended MOT file: {e}") from e

    try:
        keypoints_tracks = load_mot_tracks_from_path(tracks_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid MOT tracks file: {e}") from e

    if man.get("bounding_boxes") and keypoints_tracks:
        for tid, bb in list(man["bounding_boxes"].items()):
            if not isinstance(bb, dict):
                continue
            if bb.get("det_id") is None:
                match = _find_track_for_frame(keypoints_tracks, frame_index, bb.get("track_id", tid))
                if match:
                    bb["det_id"] = match.get("det_id")
                    bb["track_id"] = match.get("track_id", bb.get("track_id"))
    out = {
        "frame": frame_index,
        "study_id": study_id,
        "video_id": vid,
        **man,
    }
    return JSONResponse(out)


@app.post("/study/{from_study_id}/migrate/{to_study_id}")
async def migrate_study_annotations(
    request: Request,
    from_study_id: str,
    to_study_id: str
) -> JSONResponse:
    """Copy temp extended MOT annotations from one study id to another."""
    from_study_id = _safe_study_id(from_study_id)
    to_study_id = _safe_study_id(to_study_id)
    user_key, _, temp_root = _user_dirs(request)
    from_ann = _temp_annotations_txt(temp_root, from_study_id)
    to_ann = _temp_annotations_txt(temp_root, to_study_id)

    if not from_ann.exists():
        logger.warning(f"Source annotations file not found: {from_ann}")
        return JSONResponse({"status": "no_source", "message": f"No annotations found for {from_study_id}"})

    try:
        with _temp_lock(user_key, to_study_id):
            shutil.copy2(from_ann, to_ann)
        frame_annotations = _user_frame_annotations(user_key)
        if from_study_id in frame_annotations:
            if to_study_id not in frame_annotations:
                frame_annotations[to_study_id] = {}
            frame_annotations[to_study_id].update(frame_annotations[from_study_id])
        n = len(frame_annotations.get(to_study_id, {}))
        logger.info(f"Migrated temp annotations from {from_study_id} to {to_study_id} ({n} frames in memory)")
        return JSONResponse({"status": "success", "frames_migrated": n})
    except Exception as e:
        logger.error(f"Error migrating annotations: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/study/{study_id}/save")
async def save_annotations(
    request: Request,
    study_id: str,
    payload: str = Form(...)
) -> JSONResponse:
    """Save a study with all annotations."""
    if not study_id or not study_id.strip():
        raise HTTPException(status_code=400, detail="Study ID is required")

    study_id = _safe_study_id(study_id)
    
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in save_annotations: {e}, payload length: {len(payload)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    
    user_key, study_root, temp_root = _user_dirs(request)
    study_dir = study_root / study_id

    video_id = payload_data.get("video_id")
    original_filename = _safe_filename(payload_data.get("original_filename", ""))
    payload_metadata = payload_data.get("metadata", {})
    if not isinstance(payload_metadata, dict):
        payload_metadata = {}

    temp_meta: Dict[str, Any] = {}
    if video_id:
        td = _user_temp_studies(user_key).get(video_id)
        if td and isinstance(td.get("metadata"), dict):
            temp_meta = td["metadata"]
    tracks_txt_filename = payload_metadata.get("tracks_txt_filename") or temp_meta.get("tracks_txt_filename")
    if tracks_txt_filename:
        tracks_txt_filename = _safe_filename(str(tracks_txt_filename))
    else:
        tracks_txt_filename = "tracks.txt"

    source_study_id = payload_data.get("source_study_id")
    if source_study_id:
        source_study_id = _safe_study_id(source_study_id)
    is_new_upload = bool(payload_data.get("is_new_upload"))
    
    # Overwrite existing study directory if this is a new upload
    if is_new_upload and study_dir.exists():
        shutil.rmtree(study_dir, ignore_errors=True)

    # Overwrite destination for save-as if it already exists
    if source_study_id and source_study_id != study_id and study_dir.exists():
        shutil.rmtree(study_dir, ignore_errors=True)

    study_dir.mkdir(parents=True, exist_ok=True)

    # Copy original video file to study folder
    if video_id and original_filename:
        # Try to find the video file in temp directory
        video_extension = Path(original_filename).suffix.lower()
        temp_video_path = temp_root / f"video_{video_id}{video_extension}"
        study_video_path = study_dir / original_filename
        if temp_video_path.exists():
            try:
                shutil.copy2(temp_video_path, study_video_path)
                logger.info(f"Copied video file to study folder: {study_video_path}")
            except Exception as e:
                logger.warning(f"Failed to copy video file to study folder: {e}")
        elif source_study_id:
            source_path = study_root / source_study_id / original_filename
            if source_path.exists():
                try:
                    shutil.copy2(source_path, study_video_path)
                    logger.info(f"Copied video file from source study: {study_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to copy source video file: {e}")
        elif study_video_path.exists():
            logger.info("Video file already present in study folder; leaving unchanged.")
        else:
            logger.warning(f"Video file not found in temp directory: {temp_video_path}")

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required to save a study")

    temp_tracks = _temp_tracks_path(temp_root, video_id)
    study_tracks_path = study_dir / tracks_txt_filename
    try:
        if temp_tracks.exists():
            shutil.copy2(temp_tracks, study_tracks_path)
            logger.info(f"Copied MOT tracks to study: {study_tracks_path}")
        elif source_study_id:
            src_meta = load_study(study_root, source_study_id) or {}
            src_t = _study_saved_tracks_path(study_root / source_study_id, src_meta)
            if src_t and src_t.exists():
                shutil.copy2(src_t, study_tracks_path)
                logger.info(f"Copied MOT tracks from source study: {study_tracks_path}")
    except Exception as e:
        logger.warning(f"Failed to copy tracks file: {e}")

    if "keypoints_tracks" in payload_data:
        del payload_data["keypoints_tracks"]
    if "frames" in payload_data:
        del payload_data["frames"]

    ann_temp = _temp_annotations_txt(temp_root, study_id)
    try:
        with _temp_lock(user_key, study_id):
            _flush_study_memory_to_extended_mot(user_key, temp_root, study_id, video_id)
            ensure_extended_mot_copy(temp_tracks, ann_temp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.warning(f"Failed to finalize extended MOT file: {e}")

    ann_study = study_dir / f"{study_id}_annotations.txt"
    try:
        if ann_temp.exists():
            shutil.copy2(ann_temp, ann_study)
            logger.info(f"Saved annotations: {ann_study}")
    except Exception as e:
        logger.warning(f"Failed to copy annotations file: {e}")

    study_metadata = {
        "original_filename": original_filename,
        "study_id": study_id,
        "video_id": video_id,
        "video_filename": original_filename,
        "tracks_txt_filename": tracks_txt_filename,
        "tracks_source_format": "mot_txt",
    }
    metadata_path = study_dir / f"{study_id}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(study_metadata, f, indent=2)

    logger.info(f"Study {study_id} saved successfully")
    
    return JSONResponse({"status": "saved", "study_id": study_id})


def save_study(study_id: str, data: Dict[str, Any]) -> None:
    """Unused placeholder; studies are persisted via MOT-based save flow in save_annotations."""
    pass


def load_study(study_root: Path, study_id: str) -> Optional[Dict[str, Any]]:
    """Load study metadata from disk."""
    study_dir = study_root / study_id
    
    # Load metadata file (contains video_id)
    metadata_path = study_dir / f"{study_id}_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading study metadata {study_id}: {e}")
    
    return None


def _find_study_video(study_dir: Path, original_filename: str) -> Optional[Path]:
    if original_filename:
        candidate = study_dir / original_filename
        if candidate.exists():
            return candidate
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        for f in study_dir.iterdir():
            if f.suffix.lower() == ext:
                return f
    return None


def _study_data_for_propagation(
    user_key: str,
    study_root: Path,
    temp_root: Path,
    study_id: str,
    video_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    temp_data = _user_temp_studies(user_key).get(video_id) if video_id else None
    if temp_data and temp_data.get("total_frames") is not None and temp_data.get("keypoints_tracks") is not None:
        return temp_data

    study_dir = study_root / study_id
    if not study_dir.exists():
        return None

    metadata = load_study(study_root, study_id) or {}
    try:
        original_filename = _safe_filename(metadata.get("original_filename", ""))
    except HTTPException:
        original_filename = ""
    video_path = _find_study_video(study_dir, original_filename)
    if not video_path:
        return None

    try:
        video_info = get_video_info(video_path)
    except Exception:
        return None

    keypoints_tracks: Dict[Any, Any] = {}
    tracks_file = _study_saved_tracks_path(study_dir, metadata)
    if tracks_file and tracks_file.exists():
        try:
            keypoints_tracks = load_mot_tracks_from_path(tracks_file)
        except Exception:
            keypoints_tracks = {}
    if not keypoints_tracks and video_id:
        tp = _temp_tracks_path(temp_root, video_id)
        if tp.exists():
            try:
                keypoints_tracks = load_mot_tracks_from_path(tp)
            except Exception:
                keypoints_tracks = {}

    return {
        "total_frames": video_info["total_frames"],
        "keypoints_tracks": keypoints_tracks,
    }


def _find_track_for_frame(
    keypoints_tracks: Dict[int, List[Dict[str, Any]]],
    frame_index: int,
    track_id: Any,
) -> Optional[Dict[str, Any]]:
    if not keypoints_tracks or track_id is None:
        return None
    frame_tracks = keypoints_tracks.get(frame_index) or keypoints_tracks.get(str(frame_index)) or []
    for track in frame_tracks:
        track_key = track.get("track_id")
        if track_key is not None and str(track_key) == str(track_id):
            return track
    return None


def load_study_annotations_to_temp(user_key: str, study_root: Path, temp_root: Path, study_id: str) -> bool:
    """Copy saved extended MOT annotations into temp for this study session."""
    study_dir = study_root / study_id
    ann_src = study_dir / f"{study_id}_annotations.txt"
    if not ann_src.exists():
        logger.warning(f"Annotations file not found: {ann_src}")
        return False
    ann_dst = _temp_annotations_txt(temp_root, study_id)
    try:
        shutil.copy2(ann_src, ann_dst)
        logger.info(f"Copied study annotations to temp: {ann_dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy annotations to temp: {e}")
        return False


def list_studies(study_root: Path) -> List[str]:
    """List saved study IDs that have extended MOT annotation files."""
    if not study_root.exists():
        return []
    studies = []
    for d in study_root.iterdir():
        if d.is_dir() and (d / f"{d.name}_annotations.txt").exists():
            studies.append(d.name)
    return studies


@app.get("/study/{study_id}")
async def get_study(request: Request, study_id: str) -> JSONResponse:
    """
    Load a saved study: copy video and MOT tracks into temp, copy extended MOT annotations
    (or build empty extended copy from tracks), return frames and track geometry for the UI.
    """
    study_id = _safe_study_id(study_id)
    user_key, study_root, temp_root = _user_dirs(request)
    _clear_user_temp(user_key, temp_root)
    study_dir = study_root / study_id
    
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found")
    
    # Load metadata
    metadata = load_study(study_root, study_id)
    if not metadata:
        metadata = {}
    
    try:
        original_filename = _safe_filename(metadata.get("original_filename", ""))
    except HTTPException:
        original_filename = ""
    tracks_txt_filename = metadata.get("tracks_txt_filename")
    if tracks_txt_filename:
        try:
            tracks_txt_filename = _safe_filename(str(tracks_txt_filename))
        except HTTPException:
            tracks_txt_filename = ""
    if not tracks_txt_filename:
        tracks_txt_filename = "tracks.txt"

    # Find and copy video file to temp
    video_path = None
    video_id = None
    
    # Try to find video by original filename first
    if original_filename:
        potential_path = study_dir / original_filename
        if potential_path.exists():
            video_path = potential_path
    
    # If not found, look for any video file in the study directory
    if not video_path:
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            for f in study_dir.iterdir():
                if f.suffix.lower() == ext:
                    video_path = f
                    original_filename = f.name
                    break
            if video_path:
                break
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found in study folder")
    
    # Generate new video_id and copy to temp
    video_id = uuid.uuid4().hex
    video_extension = video_path.suffix.lower()
    temp_video_path = temp_root / f"video_{video_id}{video_extension}"
    
    try:
        shutil.copy2(video_path, temp_video_path)
        logger.info(f"Copied video to temp: {temp_video_path}")
    except Exception as e:
        logger.error(f"Failed to copy video to temp: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare video: {e}")
    
    # Get video info
    try:
        video_info = get_video_info(temp_video_path)
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]
        
        frames = [
            {
                "frame_index": i,
                "timestamp": i / fps if fps > 0 else 0.0,
            }
            for i in range(total_frames)
        ]
    except Exception as e:
        logger.error(f"Failed to read video info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read video: {e}")
    
    keypoints_tracks: Dict[Any, Any] = {}
    study_tracks = study_dir / tracks_txt_filename
    if not study_tracks.exists():
        study_tracks = _study_saved_tracks_path(study_dir, metadata)
    temp_tracks_path = _temp_tracks_path(temp_root, video_id)
    try:
        if study_tracks and study_tracks.exists():
            shutil.copy2(study_tracks, temp_tracks_path)
            keypoints_tracks = load_mot_tracks_from_path(temp_tracks_path)
    except Exception as e:
        logger.error(f"Failed to load MOT tracks: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid MOT tracks file in study: {e}")
    has_keypoints = len(keypoints_tracks) > 0

    load_study_annotations_to_temp(user_key, study_root, temp_root, study_id)
    ann_temp = _temp_annotations_txt(temp_root, study_id)
    if not ann_temp.exists() and temp_tracks_path.exists():
        ensure_extended_mot_copy(temp_tracks_path, ann_temp)

    tracks_meta_name = study_tracks.name if study_tracks and study_tracks.exists() else tracks_txt_filename
    meta_out = {
        "video_filename": original_filename,
        "tracks_txt_filename": tracks_meta_name,
        "total_frames": total_frames,
        "has_keypoints": has_keypoints,
        "tracks_source_format": "mot_txt",
    }

    _user_temp_studies(user_key)[video_id] = {
        "video_id": video_id,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": total_frames,
        "metadata": meta_out,
    }

    response_data = {
        "study_id": study_id,
        "video_id": video_id,
        "original_filename": original_filename,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": total_frames,
        "kind": "video",
        "metadata": meta_out,
    }
    
    return JSONResponse(response_data)


@app.get("/studies")
async def get_study_list(request: Request) -> JSONResponse:
    _, study_root, _ = _user_dirs(request)
    return JSONResponse({"studies": list_studies(study_root)})


@app.get("/study/{study_id}/export")
async def export_study_annotations(request: Request, study_id: str) -> FileResponse:
    """Export extended MOT annotations (.txt) for this study."""
    study_id = _safe_study_id(study_id)
    _, study_root, _ = _user_dirs(request)
    study_dir = study_root / study_id
    annotations_path = study_dir / f"{study_id}_annotations.txt"

    if not annotations_path.exists():
        raise HTTPException(status_code=404, detail="Study annotations file not found")

    return FileResponse(
        annotations_path,
        media_type="text/plain",
        filename=f"{study_id}_annotations.txt",
    )


@app.post("/study/{study_id}/propagate_labels")
async def propagate_labels(
    request: Request,
    study_id: str,
    payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """
    Propagate labels forward only (from frame_index onwards) for a track ID.
    Does not overwrite existing labels in earlier frames.
    payload: {track_id, frame_index, name, action, video_id}
    """
    study_id = _safe_study_id(study_id)
    track_id = payload.get("track_id")
    frame_index = payload.get("frame_index")
    name = payload.get("name")
    action = payload.get("action")
    video_id = payload.get("video_id")  # Optional: for temp studies
    
    if track_id is None or frame_index is None:
        raise HTTPException(status_code=400, detail="track_id and frame_index are required")
    
    user_key, study_root, temp_root = _user_dirs(request)
    frame_annotations = _user_frame_annotations(user_key)

    study_data = _study_data_for_propagation(user_key, study_root, temp_root, study_id, video_id)
    if not study_data:
        raise HTTPException(status_code=404, detail="Study not found. Please save the study first or provide video_id.")
    
    total_frames = study_data.get("total_frames", 0)
    keypoints_tracks = study_data.get("keypoints_tracks", {})
    
    # Propagate labels forward only (from frame_index onwards)
    updated_frames = []
    pending_updates: Dict[int, Dict[str, Any]] = {}

    def _has_nonempty_label(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        return True

    def _frame_has_existing_label(frame_ann: Dict[str, Any]) -> bool:
        if not isinstance(frame_ann, dict):
            return False
        bboxes = frame_ann.get("bounding_boxes") or {}
        bbox_data = bboxes.get(track_id)
        if bbox_data is None:
            bbox_data = bboxes.get(str(track_id))
        if not isinstance(bbox_data, dict):
            return False
        return _has_nonempty_label(bbox_data.get("name")) or _has_nonempty_label(bbox_data.get("action"))

    preexisting_labeled_frames: Set[int] = set()
    if study_id in frame_annotations:
        for frame_key, frame_ann in frame_annotations[study_id].items():
            try:
                frame_idx_int = int(frame_key)
            except Exception:
                frame_idx_int = frame_key
            if _frame_has_existing_label(frame_ann):
                preexisting_labeled_frames.add(frame_idx_int)

    ann_path = _temp_annotations_txt(temp_root, study_id)
    lines_for_scan: List[str] = []
    if ann_path.exists():
        lines_for_scan = ann_path.read_text(encoding="utf-8").splitlines()
    elif video_id:
        tracks_path = _temp_tracks_path(temp_root, video_id)
        if tracks_path.exists():
            try:
                ensure_extended_mot_copy(tracks_path, ann_path)
                lines_for_scan = ann_path.read_text(encoding="utf-8").splitlines()
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        for fi in range(total_frames):
            pl = frame_payload_from_extended_lines(lines_for_scan, fi)
            if _frame_has_existing_label(pl):
                preexisting_labeled_frames.add(fi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    tid_key = str(track_id)
    for frame_idx in range(frame_index, total_frames):
        # Check if this frame has the track_id
        frame_tracks = keypoints_tracks.get(frame_idx) or keypoints_tracks.get(str(frame_idx)) or []
        has_track = any(str(t.get("track_id")) == tid_key for t in frame_tracks)
        
        if has_track:
            if frame_idx > frame_index and frame_idx in preexisting_labeled_frames:
                break
            matched_track = _find_track_for_frame(keypoints_tracks, frame_idx, track_id)
            # Get or create annotations for this frame
            if study_id not in frame_annotations:
                frame_annotations[study_id] = {}
            if frame_idx not in frame_annotations[study_id]:
                frame_annotations[study_id][frame_idx] = {"bounding_boxes": {}}
            
            if "bounding_boxes" not in frame_annotations[study_id][frame_idx]:
                frame_annotations[study_id][frame_idx]["bounding_boxes"] = {}
            
            bboxes = frame_annotations[study_id][frame_idx]["bounding_boxes"]
            bbox_data = dict(bboxes.get(tid_key, {}))
            bbox_data.setdefault("track_id", track_id)
            if bbox_data.get("det_id") is None and matched_track:
                bbox_data["det_id"] = matched_track.get("det_id")
            
            # Always update labels for the current frame and all forward frames
            # This allows new labels to overwrite previous labels in forward frames
            # Earlier frames (before frame_index) are never touched, preserving their labels
            if name is not None and name != "":
                bbox_data["name"] = name
            elif name is not None and frame_idx == frame_index:
                # Only clear name if explicitly provided as empty on the current frame
                bbox_data.pop("name", None)
            
            if action is not None and action != "":
                bbox_data["action"] = action
            elif action is not None and frame_idx == frame_index:
                # Only clear action if explicitly provided as empty on the current frame
                bbox_data.pop("action", None)
            
            bboxes[tid_key] = bbox_data
            frame_annotations[study_id][frame_idx]["frame"] = frame_idx
            if video_id:
                frame_annotations[study_id][frame_idx]["video_id"] = video_id
            updated_frames.append(frame_idx)
            
            # Collect updates for temp file write
            annotations = {
                "frame": frame_idx,
                "keypoints": frame_annotations[study_id][frame_idx].get("keypoints", []),
                "lines": frame_annotations[study_id][frame_idx].get("lines", []),
                "rois": frame_annotations[study_id][frame_idx].get("rois", []),
                "measurements": frame_annotations[study_id][frame_idx].get("measurements", {"distances": [], "angles": []}),
                "bounding_boxes": frame_annotations[study_id][frame_idx]["bounding_boxes"],
            }
            if video_id:
                annotations["video_id"] = video_id
            pending_updates[frame_idx] = annotations

    if pending_updates and video_id:
        ann_path = _temp_annotations_txt(temp_root, study_id)
        tracks_path = _temp_tracks_path(temp_root, video_id)
        try:
            with _temp_lock(user_key, study_id):
                ensure_extended_mot_copy(tracks_path, ann_path)
                lines = ann_path.read_text(encoding="utf-8").splitlines()
                for frame_idx in sorted(pending_updates.keys()):
                    disk = frame_payload_from_extended_lines(lines, frame_idx)
                    mem = frame_annotations[study_id].get(frame_idx) or {}
                    merged = _merge_frame_payload_disk_and_memory(disk, mem)
                    sync_payload_to_extended_mot_file(ann_path, frame_idx, merged)
                    lines = ann_path.read_text(encoding="utf-8").splitlines()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.warning(f"Failed to sync propagated labels to MOT file: {e}")
    
    return JSONResponse({
        "status": "propagated",
        "updated_frames": updated_frames,
        "count": len(updated_frames)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
