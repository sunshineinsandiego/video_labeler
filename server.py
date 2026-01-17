import os
import json
import logging
import shutil
import traceback
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
STUDY_DIR = DATA_DIR / "studies"
TEMP_DIR = DATA_DIR / "temp"

# Create directories
for path in (STATIC_DIR, STUDY_DIR, TEMP_DIR):
    os.makedirs(path, exist_ok=True)

# Clear temp directory on startup
if TEMP_DIR.exists():
    try:
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        logger.info(f"Cleared temp directory: {TEMP_DIR}")
    except Exception as e:
        logger.warning(f"Error clearing temp directory: {e}")

app = FastAPI(title="Video Labeler")

# Allow local development UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/studies", StaticFiles(directory=STUDY_DIR), name="studies")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cache/clear")
async def clear_video_cache() -> JSONResponse:
    """Manually clear the video capture cache."""
    with _video_cache_lock:
        count = len(_video_cache)
        for video_id, cache_entry in _video_cache.items():
            try:
                cache_entry["capture"].release()
            except Exception:
                pass
        _video_cache.clear()
    
    logger.info(f"Cleared video cache: {count} video captures released")
    return JSONResponse({"status": "cleared", "count": count})


@app.get("/favicon.ico")
async def favicon():
    """Suppress favicon 404 errors."""
    raise HTTPException(status_code=404)


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.get("/temp/{filename:path}")
async def serve_temp_file(filename: str) -> FileResponse:
    """Serve files from temp directory."""
    file_path = TEMP_DIR / filename
    try:
        file_path = file_path.resolve()
        temp_dir_resolved = TEMP_DIR.resolve()
        if not str(file_path).startswith(str(temp_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    content_type = "application/octet-stream"
    if filename.lower().endswith(".png"):
        content_type = "image/png"
    elif filename.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif filename.lower().endswith((".mp4", ".avi", ".mov")):
        content_type = "video/mp4"
    
    return FileResponse(file_path, media_type=content_type)


def get_video_capture(video_id: str, video_path: Path) -> cv2.VideoCapture:
    """
    Get a cached video capture object, or create a new one.
    Keeps video files open in memory for faster frame access.
    """
    current_time = time.time()
    
    with _video_cache_lock:
        # Check if we have a cached capture for this video
        if video_id in _video_cache:
            cache_entry = _video_cache[video_id]
            
            # Check if capture is still valid
            if cache_entry["capture"].isOpened():
                # Update last access time
                cache_entry["last_access"] = current_time
                logger.debug(f"Reusing cached video capture for {video_id}")
                return cache_entry["capture"]
            else:
                # Capture is closed, remove from cache
                logger.info(f"Cached video capture for {video_id} was closed, removing from cache")
                del _video_cache[video_id]
        
        # Clean up stale cache entries (older than TTL)
        stale_keys = []
        for vid, cache_entry in _video_cache.items():
            if current_time - cache_entry["last_access"] > VIDEO_CACHE_TTL:
                stale_keys.append(vid)
        
        for vid in stale_keys:
            logger.info(f"Closing stale video capture for {vid} (inactive for {VIDEO_CACHE_TTL}s)")
            try:
                _video_cache[vid]["capture"].release()
            except Exception:
                pass
            del _video_cache[vid]
        
        # Create new capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Cache it
        _video_cache[video_id] = {
            "capture": cap,
            "path": video_path,
            "last_access": current_time,
        }
        
        logger.info(f"Created new cached video capture for {video_id}")
        return cap


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


def load_keypoints_tracks(jsonl_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load keypoints_tracks.jsonl file.
    Expected format: one JSON object per line with 'frame', 'track_id', 'bbox', 'keypoints', etc.
    bbox format: [x1, y1, x2, y2]
    keypoints format: [x1, y1, conf1, x2, y2, conf2, ...] (17 keypoints = 51 values)
    Returns: {frame_index: [detections]}
    """
    tracks_by_frame = {}
    
    if not jsonl_path.exists():
        logger.warning(f"Keypoints tracks file not found: {jsonl_path}")
        return tracks_by_frame
    
    logger.info(f"Loading keypoints from: {jsonl_path}")
    line_count = 0
    parsed_count = 0
    error_count = 0
    
    with jsonl_path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                data = json.loads(line)
                # Handle both 'frame' and 'frame_index' keys
                frame_index = data.get("frame", data.get("frame_index", data.get("frame_idx", 0)))
                
                if frame_index not in tracks_by_frame:
                    tracks_by_frame[frame_index] = []
                
                tracks_by_frame[frame_index].append(data)
                parsed_count += 1
                
                # Log first few tracks for debugging
                if parsed_count <= 3:
                    logger.info(f"Parsed track {parsed_count}: frame={frame_index}, track_id={data.get('track_id')}, has_bbox={bool(data.get('bbox'))}, has_keypoints={bool(data.get('keypoints'))}")
            except json.JSONDecodeError as e:
                error_count += 1
                logger.warning(f"Error parsing line {line_num} in {jsonl_path}: {e}")
                continue
    
    logger.info(f"Loaded keypoints: {line_count} total lines, {parsed_count} parsed successfully, {error_count} errors, {len(tracks_by_frame)} unique frames")
    return tracks_by_frame


@app.post("/upload")
async def upload_file(
    video_file: UploadFile = File(...),
    keypoints_file: UploadFile = File(None)
) -> JSONResponse:
    logger.info(f"Upload request received: video_file={video_file.filename}, keypoints_file={keypoints_file.filename if keypoints_file else None}")
    """
    Upload a video file and optional keypoints_tracks.jsonl file.
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Video filename is required.")
    
    # Save video file
    video_extension = Path(video_file.filename).suffix.lower()
    if video_extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    
    video_id = uuid.uuid4().hex
    video_path = TEMP_DIR / f"video_{video_id}{video_extension}"
    
    try:
        with video_path.open("wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store video: {exc}") from exc
    
    # Get video metadata (without extracting frames - we'll stream them on-demand)
    try:
        video_info = get_video_info(video_path)
        total_frames = video_info["total_frames"]
        if total_frames == 0:
            raise HTTPException(status_code=400, detail="Video has no frames.")
        
        # Create frame list metadata (without extracting actual frames)
        frames = [
            {
                "frame_index": i,
                "timestamp": i / video_info["fps"] if video_info["fps"] > 0 else 0.0,
            }
            for i in range(total_frames)
        ]
    except Exception as exc:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to read video: {exc}") from exc
    
    # Load keypoints tracks if provided
    keypoints_tracks = {}
    logger.info(f"Checking keypoints_file: {keypoints_file}, filename: {keypoints_file.filename if keypoints_file else None}")
    
    if keypoints_file is not None and keypoints_file.filename:
        keypoints_path = TEMP_DIR / f"keypoints_{video_id}.jsonl"
        try:
            file_size = keypoints_file.size if hasattr(keypoints_file, 'size') else 'unknown'
            logger.info(f"Received keypoints file: {keypoints_file.filename}, size: {file_size}")
            
            with keypoints_path.open("wb") as buffer:
                content = await keypoints_file.read()
                buffer.write(content)
            
            logger.info(f"Saved keypoints file to: {keypoints_path}, size: {len(content)} bytes")
            
            if len(content) == 0:
                logger.warning(f"Keypoints file is empty!")
            else:
                # Verify file was written correctly
                if not keypoints_path.exists() or keypoints_path.stat().st_size == 0:
                    logger.error(f"Keypoints file was not saved correctly!")
                else:
                    keypoints_tracks = load_keypoints_tracks(keypoints_path)
                    logger.info(f"Loaded keypoints tracks: {len(keypoints_tracks)} frames with tracks")
                    if len(keypoints_tracks) > 0:
                        sample_frame = list(keypoints_tracks.keys())[0]
                        logger.info(f"Sample frame {sample_frame} has {len(keypoints_tracks[sample_frame])} tracks")
                    else:
                        logger.warning(f"No tracks loaded from keypoints file - file may be empty or malformed")
        except Exception as exc:
            logger.error(f"Error loading keypoints tracks: {exc}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info("No keypoints file provided in upload (keypoints_file is None or has no filename)")
    
    # Get first frame image URL (now streams from video file)
    first_image_url = f"/video/{video_id}/frame/0"
    
    response_data = {
        "video_id": video_id,
        "image_url": first_image_url,
        "stored_filename": video_path.name,
        "kind": "video",
        "frames": frames,  # Frame metadata (indices and timestamps) without actual frame images
        "keypoints_tracks": keypoints_tracks,
        "total_frames": len(frames),
        "metadata": {
            "video_filename": video_file.filename,
            "total_frames": len(frames),
            "has_keypoints": len(keypoints_tracks) > 0,
        },
    }
    
    # Store video metadata for potential temp study use
    # This allows propagation to work even before study is saved
    _temp_studies[video_id] = {
        "video_id": video_id,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": len(frames),
        "metadata": response_data["metadata"],
    }
    
    return JSONResponse(response_data)


@app.get("/video/{video_id}/frame/{frame_index}")
async def get_frame(video_id: str, frame_index: int) -> Response:
    """Get a specific frame image by streaming from video file (with caching)."""
    # Find the video file
    video_path = None
    
    # Try common video extensions
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        potential_path = TEMP_DIR / f"video_{video_id}{ext}"
        if potential_path.exists():
            video_path = potential_path
            break
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found for video_id: {video_id}")
    
    try:
        # Get cached video capture (or create new one)
        # This keeps the video file open in SERVER memory, not browser
        cap = get_video_capture(video_id, video_path)
        
        # Seek to the requested frame
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
        
    except Exception as e:
        logger.error(f"Error extracting frame {frame_index} from video {video_id}: {e}")
        # On error, try to remove from cache and retry once
        with _video_cache_lock:
            if video_id in _video_cache:
                try:
                    _video_cache[video_id]["capture"].release()
                except Exception:
                    pass
                del _video_cache[video_id]
        raise HTTPException(status_code=500, detail=f"Failed to extract frame: {str(e)}")


@app.get("/video/{video_id}/keypoints/{frame_index}")
async def get_frame_keypoints(video_id: str, frame_index: int) -> JSONResponse:
    """Get keypoints/tracks for a specific frame."""
    # Try to load from temp file
    keypoints_path = TEMP_DIR / f"keypoints_{video_id}.jsonl"
    logger.info(f"Requesting keypoints for video {video_id}, frame {frame_index}")
    logger.info(f"Keypoints file path: {keypoints_path}, exists: {keypoints_path.exists()}")
    
    if keypoints_path.exists():
        keypoints_tracks = load_keypoints_tracks(keypoints_path)
        logger.info(f"Loaded {len(keypoints_tracks)} frames from keypoints file for video {video_id}")
        
        # Handle both 'frame' and 'frame_index' keys in the data
        frame_tracks = keypoints_tracks.get(frame_index, [])
        logger.info(f"Found {len(frame_tracks)} tracks for frame {frame_index}")
        return JSONResponse({"tracks": frame_tracks})
    else:
        # No keypoints file - return empty tracks (no warning, this is normal)
        return JSONResponse({"tracks": []})


# Store annotations per frame in memory
_frame_annotations: Dict[str, Dict[int, Dict[str, Any]]] = {}

# Store video metadata (keypoints_tracks, frames, etc.) for temp studies
_temp_studies: Dict[str, Dict[str, Any]] = {}

# Video capture cache to avoid reopening video files repeatedly
# Keeps video files open in SERVER memory for faster frame access
_video_cache: Dict[str, Dict[str, Any]] = {}  # {video_id: {"capture": cv2.VideoCapture, "path": Path, "last_access": time.time()}}
_video_cache_lock = Lock()
VIDEO_CACHE_TTL = 300  # 5 minutes in seconds


@app.post("/study/{study_id}/frame/{frame_index}/annotations")
async def save_frame_annotations(
    study_id: str,
    frame_index: int,
    payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Save annotations for a specific frame."""
    if study_id not in _frame_annotations:
        _frame_annotations[study_id] = {}
    
    _frame_annotations[study_id][frame_index] = payload
    
    # Save to temp file for persistence
    temp_annotations_file = TEMP_DIR / f"{study_id}_annotations.json"
    try:
        # Load existing annotations first
        existing_annotations = {}
        if temp_annotations_file.exists():
            try:
                with temp_annotations_file.open("r", encoding="utf-8") as f:
                    existing_annotations = json.load(f)
            except Exception:
                existing_annotations = {}
        
        # Update with current study's annotations
        if study_id not in existing_annotations:
            existing_annotations[study_id] = {}
        
        # Convert frame_index to string for JSON consistency
        existing_annotations[study_id][str(frame_index)] = payload
        
        # Write back
        with temp_annotations_file.open("w", encoding="utf-8") as f:
            json.dump(existing_annotations, f, indent=2)
        
        # Log what was saved
        keypoints_count = len(payload.get('keypoints', []))
        lines_count = len(payload.get('lines', []))
        rois_count = len(payload.get('rois', []))
        distances_count = len(payload.get('measurements', {}).get('distances', []))
        angles_count = len(payload.get('measurements', {}).get('angles', []))
        logger.info(f"Saved annotations for frame {frame_index} of study {study_id}: keypoints={keypoints_count}, lines={lines_count}, rois={rois_count}, distances={distances_count}, angles={angles_count}")
        
        # Also update in-memory storage
        if study_id not in _frame_annotations:
            _frame_annotations[study_id] = {}
        _frame_annotations[study_id][frame_index] = payload
    except Exception as e:
        logger.warning(f"Failed to save temp annotations file: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return JSONResponse({"status": "saved"})


@app.get("/study/{study_id}/frame/{frame_index}/annotations")
async def get_frame_annotations(
    study_id: str,
    frame_index: int
) -> JSONResponse:
    """Get annotations for a specific frame."""

    # 1. Check Memory (Fastest, most recent)
    frame_annotations = _frame_annotations.get(study_id, {})
    # Check int key then str key
    annotations = frame_annotations.get(frame_index) or frame_annotations.get(str(frame_index))
    
    if annotations:
        return JSONResponse(annotations)

    # 2. Check Temp File (Active session persistence)
    temp_annotations_file = TEMP_DIR / f"{study_id}_annotations.json"
    if temp_annotations_file.exists():
        try:
            with temp_annotations_file.open("r", encoding="utf-8") as f:
                loaded_annotations = json.load(f)
                if study_id in loaded_annotations:
                    frame_ann = loaded_annotations[study_id]
                    annotations = frame_ann.get(str(frame_index)) or frame_ann.get(frame_index)
                    if annotations:
                        return JSONResponse(annotations)
        except Exception:
            pass

    """Get annotations for a specific frame."""
    # Try to load from saved study file first
    study_data = load_study(study_id)
    if study_data and "frame_annotations" in study_data:
        frame_annotations = study_data["frame_annotations"]
        annotations = frame_annotations.get(str(frame_index)) or frame_annotations.get(frame_index)
        if annotations:
            return JSONResponse(annotations)
    
 
    if not annotations:
        annotations = {
            "bounding_boxes": {},  # {track_id: {bbox, name, action, annotations}}
            "keypoints": [],
            "lines": [],
            "rois": [],
            "measurements": {"distances": [], "angles": []},
        }
    
    return JSONResponse(annotations)


@app.post("/study/{from_study_id}/migrate/{to_study_id}")
async def migrate_study_annotations(from_study_id: str, to_study_id: str) -> JSONResponse:
    """Migrate annotations from one study ID to another in the temp file."""
    from_temp_file = TEMP_DIR / f"{from_study_id}_annotations.json"
    to_temp_file = TEMP_DIR / f"{to_study_id}_annotations.json"
    
    if not from_temp_file.exists():
        logger.warning(f"Source temp file not found: {from_temp_file}")
        return JSONResponse({"status": "no_source", "message": f"No annotations found for {from_study_id}"})
    
    try:
        # Load source annotations
        with from_temp_file.open("r", encoding="utf-8") as f:
            temp_annotations = json.load(f)
        
        # Get source study data
        if from_study_id not in temp_annotations:
            logger.warning(f"Source study {from_study_id} not found in temp file")
            return JSONResponse({"status": "no_data", "message": f"No annotations found for {from_study_id}"})
        
        source_data = temp_annotations[from_study_id]
        
        # Load or create destination file
        if to_temp_file.exists():
            with to_temp_file.open("r", encoding="utf-8") as f:
                dest_annotations = json.load(f)
        else:
            dest_annotations = {}
        
        # Copy annotations to destination
        if to_study_id not in dest_annotations:
            dest_annotations[to_study_id] = {}
        
        # Merge source into destination (destination takes precedence for conflicts)
        for frame_idx, frame_data in source_data.items():
            if frame_idx not in dest_annotations[to_study_id]:
                dest_annotations[to_study_id][frame_idx] = frame_data
            else:
                # Merge: combine arrays, prefer destination for conflicts
                dest_frame = dest_annotations[to_study_id][frame_idx]
                # Merge keypoints, lines, etc. (append if not already present)
                for key in ['keypoints', 'lines', 'rois']:
                    if key in frame_data and isinstance(frame_data[key], list):
                        existing_ids = {item.get('id') for item in dest_frame.get(key, [])}
                        for item in frame_data[key]:
                            if item.get('id') not in existing_ids:
                                if key not in dest_frame:
                                    dest_frame[key] = []
                                dest_frame[key].append(item)
                
                # Merge measurements
                if 'measurements' in frame_data:
                    if 'measurements' not in dest_frame:
                        dest_frame['measurements'] = {'distances': [], 'angles': []}
                    for mtype in ['distances', 'angles']:
                        if mtype in frame_data['measurements']:
                            existing_ids = {item.get('id') for item in dest_frame['measurements'].get(mtype, [])}
                            for item in frame_data['measurements'][mtype]:
                                if item.get('id') not in existing_ids:
                                    dest_frame['measurements'][mtype].append(item)
                
                # Merge bounding boxes
                if 'bounding_boxes' in frame_data:
                    if 'bounding_boxes' not in dest_frame:
                        dest_frame['bounding_boxes'] = {}
                    dest_frame['bounding_boxes'].update(frame_data['bounding_boxes'])
        
        # Save destination file
        with to_temp_file.open("w", encoding="utf-8") as f:
            json.dump(dest_annotations, f, indent=2)
        
        # Also update in-memory storage
        if from_study_id in _frame_annotations:
            if to_study_id not in _frame_annotations:
                _frame_annotations[to_study_id] = {}
            _frame_annotations[to_study_id].update(_frame_annotations[from_study_id])
        
        logger.info(f"Migrated annotations from {from_study_id} to {to_study_id}: {len(source_data)} frames")
        return JSONResponse({"status": "success", "frames_migrated": len(source_data)})
    
    except Exception as e:
        logger.error(f"Error migrating annotations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/study/{study_id}/save")
async def save_annotations(
    study_id: str,
    payload: str = Form(...)
) -> JSONResponse:
    """Save a study with all annotations."""
    if not study_id or not study_id.strip():
        raise HTTPException(status_code=400, detail="Study ID is required")
    
    study_id = study_id.strip()
    
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in save_annotations: {e}, payload length: {len(payload)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    
    study_dir = STUDY_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video_id and original filename from payload
    video_id = payload_data.get("video_id")
    original_filename = payload_data.get("original_filename", "")
    
    # Copy original video file to study folder
    if video_id and original_filename:
        # Try to find the video file in temp directory
        video_extension = Path(original_filename).suffix.lower()
        temp_video_path = TEMP_DIR / f"video_{video_id}{video_extension}"
        if temp_video_path.exists():
            study_video_path = study_dir / original_filename
            try:
                shutil.copy2(temp_video_path, study_video_path)
                logger.info(f"Copied video file to study folder: {study_video_path}")
            except Exception as e:
                logger.warning(f"Failed to copy video file to study folder: {e}")
        else:
            logger.warning(f"Video file not found in temp directory: {temp_video_path}")
    
    # Remove keypoints_tracks from payload_data - we don't need to save it
    # It can be loaded from the temp file when needed (for export)
    if "keypoints_tracks" in payload_data:
        del payload_data["keypoints_tracks"]
    
    # Also remove frames array - we don't need to save it, can be regenerated
    if "frames" in payload_data:
        del payload_data["frames"]
    
    # Collect all frame annotations from temp file (this is the source of truth)
    # The temp file contains all annotations created while navigating frames
    all_frame_annotations = {}
    
    # Load from temp file (this is where annotations are persisted during navigation)
    temp_annotations_file = TEMP_DIR / f"{study_id}_annotations.json"
    if temp_annotations_file.exists():
        try:
            with temp_annotations_file.open("r", encoding="utf-8") as f:
                temp_annotations = json.load(f)
                logger.info(f"Loaded temp annotations file: {list(temp_annotations.keys())}")
                if study_id in temp_annotations:
                    frame_data = temp_annotations[study_id]
                    logger.info(f"Found {len(frame_data)} frames in temp file for study {study_id}, keys: {list(frame_data.keys())[:10]}")
                    # Convert string keys to int keys
                    for key, value in frame_data.items():
                        frame_idx = int(key) if isinstance(key, str) and key.isdigit() else key
                        all_frame_annotations[frame_idx] = value
                        # Log first frame's structure and content
                        if len(all_frame_annotations) == 1:
                            logger.info(f"Sample frame {frame_idx} annotations keys: {list(value.keys())}")
                            # Log actual content counts
                            kp_count = len(value.get('keypoints', []))
                            lines_count = len(value.get('lines', []))
                            rois_count = len(value.get('rois', []))
                            dist_count = len(value.get('measurements', {}).get('distances', []))
                            ang_count = len(value.get('measurements', {}).get('angles', []))
                            logger.info(f"Sample frame {frame_idx} content: keypoints={kp_count}, lines={lines_count}, rois={rois_count}, distances={dist_count}, angles={ang_count}")
                else:
                    logger.warning(f"Study {study_id} not found in temp annotations file. Available studies: {list(temp_annotations.keys())}")
        except Exception as e:
            logger.warning(f"Error loading temp annotations file: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info(f"Temp annotations file not found: {temp_annotations_file}")
    
    # Merge in-memory annotations (in case temp file is out of sync)
    if study_id in _frame_annotations:
        logger.info(f"Found {len(_frame_annotations[study_id])} frames in memory for study {study_id}")
        for frame_idx, annotations in _frame_annotations[study_id].items():
            # Only merge if frame not already in all_frame_annotations, or if memory has more recent data
            if frame_idx not in all_frame_annotations:
                all_frame_annotations[frame_idx] = annotations
    
    logger.info(f"Total frames with annotations after loading: {len(all_frame_annotations)}")
    
    # Build map of track_id -> annotations
    track_annotations = {}
    annotated_track_ids = set()
    
    for frame_idx, frame_ann in all_frame_annotations.items():
        bounding_boxes = frame_ann.get("bounding_boxes", {})
        for track_id_str, bbox_ann in bounding_boxes.items():
            if bbox_ann.get("name") or bbox_ann.get("action"):
                try:
                    track_id = int(track_id_str) if isinstance(track_id_str, str) and track_id_str.isdigit() else track_id_str
                    annotated_track_ids.add(track_id)
                    track_annotations[track_id] = {
                        "name": bbox_ann.get("name"),
                        "action": bbox_ann.get("action")
                    }
                except (ValueError, TypeError):
                    annotated_track_ids.add(track_id_str)
                    track_annotations[track_id_str] = {
                        "name": bbox_ann.get("name"),
                        "action": bbox_ann.get("action")
                    }
    
    # Save as JSONL file
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required to save annotations")
    
    keypoints_path = TEMP_DIR / f"keypoints_{video_id}.jsonl"
    annotations_path = study_dir / f"{study_id}_annotations.jsonl"
    
    # Helper function to serialize annotation objects with full data for reconstruction
    def serialize_keypoint(kp):
        return {
            "id": kp.get("id", ""),
            "name": kp.get("label", ""),
            "x": kp.get("x", 0),
            "y": kp.get("y", 0),
            "color": kp.get("color", "#ff5252")
        }
    
    def serialize_line(line):
        return {
            "id": line.get("id", ""),
            "name": line.get("label", ""),
            "start": {"x": line.get("start", {}).get("x", 0), "y": line.get("start", {}).get("y", 0)},
            "end": {"x": line.get("end", {}).get("x", 0), "y": line.get("end", {}).get("y", 0)},
            "color": line.get("color", "#4fc3f7")
        }
    
    def serialize_roi(roi):
        return {
            "id": roi.get("id", ""),
            "name": roi.get("label", ""),
            "points": [{"x": p.get("x", 0), "y": p.get("y", 0)} for p in roi.get("points", [])],
            "color": roi.get("color", "#4fc3f7")
        }
    
    def serialize_distance(dist):
        return {
            "id": dist.get("id", ""),
            "name": dist.get("label", ""),
            "start": {"x": dist.get("start", {}).get("x", 0), "y": dist.get("start", {}).get("y", 0)},
            "end": {"x": dist.get("end", {}).get("x", 0), "y": dist.get("end", {}).get("y", 0)},
            "value": dist.get("value", 0),
            "color": dist.get("color", "#4fc3f7")
        }
    
    def serialize_angle(angle):
        return {
            "id": angle.get("id", ""),
            "name": angle.get("label", ""),
            "point1": {"x": angle.get("point1", {}).get("x", 0), "y": angle.get("point1", {}).get("y", 0)},
            "point2": {"x": angle.get("point2", {}).get("x", 0), "y": angle.get("point2", {}).get("y", 0)},
            "point3": {"x": angle.get("point3", {}).get("x", 0), "y": angle.get("point3", {}).get("y", 0)},
            "value": angle.get("value", 0),
            "isReflex": angle.get("isReflex", False),
            "color": angle.get("color", "#4fc3f7")
        }
    
    if keypoints_path.exists():
        # Case 1: Has keypoints_tracks file - save annotated tracks with name/action
        # Manual annotations are saved as separate frame-level records (no track_id)
        lines_written = 0
        frames_with_manual_annotations_written = set()  # Track which frames already have manual annotations written
        
        with keypoints_path.open("r", encoding="utf-8") as input_file, \
             annotations_path.open("w", encoding="utf-8") as output_file:
            
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    track_record = json.loads(line)
                    track_id = track_record.get("track_id")
                    frame_idx = track_record.get("frame")
                    
                    # Only include annotated tracks
                    if track_id is None or track_id not in annotated_track_ids:
                        continue
                    
                    # Get name and action
                    ann = track_annotations.get(track_id)
                    name_val = None
                    action_val = None
                    if ann:
                        name_val = ann.get("name")
                        if name_val and name_val.strip():
                            name_val = name_val.strip()
                        else:
                            name_val = None
                        
                        action_val = ann.get("action")
                        if action_val and action_val.strip():
                            action_val = action_val.strip()
                        else:
                            action_val = None
                    
                    # Add name and action if present
                    if name_val:
                        track_record["name"] = name_val
                    if action_val:
                        track_record["action"] = action_val
                    
                    # Write the track record (without manual annotations - those go in separate frame records)
                    output_file.write(json.dumps(track_record, ensure_ascii=False) + "\n")
                    lines_written += 1
                    
                    # Write manual annotations for this frame as a separate record (only once per frame)
                    if frame_idx not in frames_with_manual_annotations_written:
                        frame_ann = all_frame_annotations.get(frame_idx) or all_frame_annotations.get(str(frame_idx), {})
                        
                        keypoints_list = frame_ann.get("keypoints", [])
                        lines_list = frame_ann.get("lines", [])
                        rois_list = frame_ann.get("rois", [])
                        measurements = frame_ann.get("measurements", {})
                        distances_list = measurements.get("distances", []) if isinstance(measurements, dict) else []
                        angles_list = measurements.get("angles", []) if isinstance(measurements, dict) else []
                        
                        has_any_manual = keypoints_list or lines_list or rois_list or distances_list or angles_list
                        
                        if has_any_manual:
                            # Frame-level record (no track_id) for manual annotations
                            frame_record = {
                                "frame": frame_idx,
                                "Keypoints": [serialize_keypoint(kp) for kp in keypoints_list],
                                "Lines": [serialize_line(ln) for ln in lines_list],
                                "ROIs": [serialize_roi(roi) for roi in rois_list],
                                "Distances": [serialize_distance(dist) for dist in distances_list],
                                "Angles": [serialize_angle(ang) for ang in angles_list]
                            }
                            output_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
                            lines_written += 1
                        
                        frames_with_manual_annotations_written.add(frame_idx)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line in {keypoints_path}: {e}")
                    continue
        
        # Also write manual annotations for frames that don't have any tracks
        # (in case user added manual annotations on frames without annotated tracks)
        with annotations_path.open("a", encoding="utf-8") as output_file:
            for frame_idx in all_frame_annotations.keys():
                # Normalize frame_idx to int
                if isinstance(frame_idx, str) and frame_idx.isdigit():
                    frame_idx_int = int(frame_idx)
                else:
                    frame_idx_int = frame_idx
                
                if frame_idx_int in frames_with_manual_annotations_written:
                    continue
                
                frame_ann = all_frame_annotations.get(frame_idx) or all_frame_annotations.get(str(frame_idx), {})
                
                keypoints_list = frame_ann.get("keypoints", [])
                lines_list = frame_ann.get("lines", [])
                rois_list = frame_ann.get("rois", [])
                measurements = frame_ann.get("measurements", {})
                distances_list = measurements.get("distances", []) if isinstance(measurements, dict) else []
                angles_list = measurements.get("angles", []) if isinstance(measurements, dict) else []
                
                has_any_manual = keypoints_list or lines_list or rois_list or distances_list or angles_list
                
                if has_any_manual:
                    frame_record = {
                        "frame": frame_idx_int,
                        "Keypoints": [serialize_keypoint(kp) for kp in keypoints_list],
                        "Lines": [serialize_line(ln) for ln in lines_list],
                        "ROIs": [serialize_roi(roi) for roi in rois_list],
                        "Distances": [serialize_distance(dist) for dist in distances_list],
                        "Angles": [serialize_angle(ang) for ang in angles_list]
                    }
                    output_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
                    lines_written += 1
                    frames_with_manual_annotations_written.add(frame_idx_int)
        
        logger.info(f"Saved study {study_id} as JSONL: {lines_written} lines written (includes {len(frames_with_manual_annotations_written)} frame annotation records)")
        
        # Also save metadata file with original video filename
        metadata_path = study_dir / f"{study_id}_metadata.json"
        original_filename = payload_data.get("original_filename", "")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump({"original_filename": original_filename, "study_id": study_id}, f, indent=2)
    else:
        # Case 2: No keypoints_tracks file - save frame annotations (keypoints, lines, ROIs, distances, angles)
        lines_written = 0
        
        logger.info(f"Saving annotations for study {study_id} (no keypoints file). Found {len(all_frame_annotations)} frames with annotations data.")
        
        with annotations_path.open("w", encoding="utf-8") as output_file:
            # Write one line per frame that has annotations
            # Check both integer and string keys for frame indices
            all_frame_indices = set()
            for key in all_frame_annotations.keys():
                if isinstance(key, str) and key.isdigit():
                    all_frame_indices.add(int(key))
                elif isinstance(key, int):
                    all_frame_indices.add(key)
                else:
                    all_frame_indices.add(key)
            
            logger.info(f"Frame indices found: {sorted(all_frame_indices)}")
            
            for frame_idx in sorted(all_frame_indices):
                # Try both integer and string key
                frame_ann = all_frame_annotations.get(frame_idx) or all_frame_annotations.get(str(frame_idx), {})
                
                logger.info(f"Processing frame {frame_idx}: {list(frame_ann.keys())}")
                
                # Skip frames with no annotations
                keypoints_list = frame_ann.get("keypoints", [])
                lines_list = frame_ann.get("lines", [])
                rois_list = frame_ann.get("rois", [])
                measurements = frame_ann.get("measurements", {})
                distances_list = measurements.get("distances", []) if isinstance(measurements, dict) else []
                angles_list = measurements.get("angles", []) if isinstance(measurements, dict) else []
                
                has_keypoints = len(keypoints_list) > 0
                has_lines = len(lines_list) > 0
                has_rois = len(rois_list) > 0
                has_distances = len(distances_list) > 0
                has_angles = len(angles_list) > 0
                
                logger.info(f"Frame {frame_idx}: keypoints={has_keypoints} ({len(keypoints_list)}), lines={has_lines} ({len(lines_list)}), rois={has_rois} ({len(rois_list)}), distances={has_distances} ({len(distances_list)}), angles={has_angles} ({len(angles_list)})")
                
                if not (has_keypoints or has_lines or has_rois or has_distances or has_angles):
                    continue
                
                # Build frame record with full annotation data for reconstruction
                frame_record = {
                    "frame": frame_idx,
                    "Keypoints": [serialize_keypoint(kp) for kp in keypoints_list],
                    "Lines": [serialize_line(ln) for ln in lines_list],
                    "ROIs": [serialize_roi(roi) for roi in rois_list],
                    "Distances": [serialize_distance(dist) for dist in distances_list],
                    "Angles": [serialize_angle(ang) for ang in angles_list]
                }
                
                # Write the frame record
                output_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
                lines_written += 1
        
        logger.info(f"Saved study {study_id} as JSONL (no keypoints file): {lines_written} frames with annotations written")
    
    # Also save metadata file with original video filename
    metadata_path = study_dir / f"{study_id}_metadata.json"
    original_filename = payload_data.get("original_filename", "")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"original_filename": original_filename, "study_id": study_id}, f, indent=2)
    
    logger.info(f"Study {study_id} saved successfully")
    
    return JSONResponse({"status": "saved", "study_id": study_id})


def save_study(study_id: str, data: Dict[str, Any]) -> None:
    """Save study data to disk. This function is no longer used - annotations are saved as JSONL."""
    # This function is kept for backward compatibility but annotations are now saved as JSONL
    pass


def load_study(study_id: str) -> Optional[Dict[str, Any]]:
    """Load study metadata from disk."""
    study_dir = STUDY_DIR / study_id
    
    # Load metadata file (contains video_id)
    metadata_path = study_dir / f"{study_id}_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading study metadata {study_id}: {e}")
    
    # Try old JSON format for backward compatibility
    annotations_path = study_dir / f"{study_id}_annotations.json"
    if annotations_path.exists():
        try:
            with annotations_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading study {study_id}: {e}")
            return None
    
    return None


def parse_coords(coords_str: str) -> List[Dict[str, float]]:
    """Parse coordinate string like 'x1,y1;x2,y2;...' into list of {x, y} dicts."""
    points = []
    if not coords_str:
        return points
    for pair in coords_str.split(";"):
        parts = pair.split(",")
        if len(parts) >= 2:
            try:
                points.append({"x": float(parts[0]), "y": float(parts[1])})
            except ValueError:
                continue
    return points


def load_study_annotations_to_temp(study_id: str) -> bool:
    """
    Load annotations from saved JSONL file and write to temp JSON format.
    Converts from saved format (Keypoints, coords string) to temp format (keypoints, x/y fields).
    Returns True if successful, False otherwise.
    """
    study_dir = STUDY_DIR / study_id
    annotations_path = study_dir / f"{study_id}_annotations.jsonl"
    temp_annotations_file = TEMP_DIR / f"{study_id}_annotations.json"
    
    if not annotations_path.exists():
        logger.warning(f"Annotations file not found: {annotations_path}")
        return False
    
    logger.info(f"Loading annotations from {annotations_path} to temp")
    
    all_frame_annotations = {}
    
    try:
        with annotations_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    frame_idx = record.get("frame", 0)
                    
                    # Skip track records (they have track_id), only process frame annotation records
                    if "track_id" in record:
                        # This is a track record with bounding box labels
                        # We can extract name/action for bounding_boxes
                        track_id = record.get("track_id")
                        if frame_idx not in all_frame_annotations:
                            all_frame_annotations[frame_idx] = {
                                "keypoints": [],
                                "lines": [],
                                "rois": [],
                                "measurements": {"distances": [], "angles": []},
                                "bounding_boxes": {}
                            }
                        
                        if track_id is not None:
                            name = record.get("name")
                            action = record.get("action")
                            if name or action:
                                all_frame_annotations[frame_idx]["bounding_boxes"][str(track_id)] = {
                                    "name": name,
                                    "action": action
                                }
                        continue
                    
                    # This is a frame annotation record (manual annotations)
                    if frame_idx not in all_frame_annotations:
                        all_frame_annotations[frame_idx] = {
                            "keypoints": [],
                            "lines": [],
                            "rois": [],
                            "measurements": {"distances": [], "angles": []},
                            "bounding_boxes": {}
                        }
                    
                    frame_ann = all_frame_annotations[frame_idx]
                    
                    # Parse Keypoints (handles both old coords format and new x/y format)
                    keypoints_data = record.get("Keypoints", [])
                    for i, kp in enumerate(keypoints_data):
                        # Try new format first (individual x, y fields)
                        if "x" in kp and "y" in kp:
                            frame_ann["keypoints"].append({
                                "id": kp.get("id", f"kp-{i+1}"),
                                "label": kp.get("name", kp.get("label", f"#{i+1}")),
                                "x": kp.get("x", 0),
                                "y": kp.get("y", 0),
                                "color": kp.get("color", "#ff5252")
                            })
                        # Fall back to old coords string format
                        elif "coords" in kp:
                            coords = parse_coords(kp.get("coords", ""))
                            if coords:
                                frame_ann["keypoints"].append({
                                    "id": kp.get("id", f"kp-{i+1}"),
                                    "label": kp.get("name", f"#{i+1}"),
                                    "x": coords[0]["x"],
                                    "y": coords[0]["y"],
                                    "color": kp.get("color", "#ff5252")
                                })
                    
                    # Parse Lines (handles both old coords format and new start/end format)
                    lines_data = record.get("Lines", [])
                    for i, ln in enumerate(lines_data):
                        # Try new format first (start/end objects)
                        if "start" in ln and "end" in ln:
                            frame_ann["lines"].append({
                                "id": ln.get("id", f"line-{i+1}"),
                                "label": ln.get("name", ln.get("label", f"#{i+1}")),
                                "start": ln.get("start"),
                                "end": ln.get("end"),
                                "color": ln.get("color", "#4fc3f7")
                            })
                        # Fall back to old coords string format
                        elif "coords" in ln:
                            coords = parse_coords(ln.get("coords", ""))
                            if len(coords) >= 2:
                                frame_ann["lines"].append({
                                    "id": ln.get("id", f"line-{i+1}"),
                                    "label": ln.get("name", f"#{i+1}"),
                                    "start": coords[0],
                                    "end": coords[1],
                                    "color": ln.get("color", "#4fc3f7")
                                })
                    
                    # Parse ROIs (handles both old coords format and new points format)
                    rois_data = record.get("ROIs", [])
                    for i, roi in enumerate(rois_data):
                        # Try new format first (points array)
                        if "points" in roi and isinstance(roi.get("points"), list):
                            frame_ann["rois"].append({
                                "id": roi.get("id", f"roi-{i+1}"),
                                "label": roi.get("name", roi.get("label", f"#{i+1}")),
                                "points": roi.get("points"),
                                "color": roi.get("color", "#4fc3f7")
                            })
                        # Fall back to old coords string format
                        elif "coords" in roi:
                            coords = parse_coords(roi.get("coords", ""))
                            if len(coords) >= 3:
                                frame_ann["rois"].append({
                                    "id": roi.get("id", f"roi-{i+1}"),
                                    "label": roi.get("name", f"#{i+1}"),
                                    "points": coords,
                                    "color": roi.get("color", "#4fc3f7")
                                })
                    
                    # Parse Distances (handles both old coords format and new start/end format)
                    distances_data = record.get("Distances", [])
                    for i, dist in enumerate(distances_data):
                        start_pt = None
                        end_pt = None
                        
                        # Try new format first (start/end objects)
                        if "start" in dist and "end" in dist:
                            start_pt = dist.get("start")
                            end_pt = dist.get("end")
                        # Fall back to old coords string format
                        elif "coords" in dist:
                            coords = parse_coords(dist.get("coords", ""))
                            if len(coords) >= 2:
                                start_pt = coords[0]
                                end_pt = coords[1]
                        
                        if start_pt and end_pt:
                            value = dist.get("value")
                            if value is None:
                                # Calculate distance if not provided
                                dx = end_pt["x"] - start_pt["x"]
                                dy = end_pt["y"] - start_pt["y"]
                                value = (dx**2 + dy**2) ** 0.5
                            frame_ann["measurements"]["distances"].append({
                                "id": dist.get("id", f"dist-{i+1}"),
                                "label": dist.get("name", dist.get("label", f"#{i+1}")),
                                "start": start_pt,
                                "end": end_pt,
                                "value": value,
                                "color": dist.get("color", "#4fc3f7")
                            })
                    
                    # Parse Angles (handles both old coords format and new point1/point2/point3 format)
                    angles_data = record.get("Angles", [])
                    for i, ang in enumerate(angles_data):
                        p1 = None
                        p2 = None
                        p3 = None
                        
                        # Try new format first (point1/point2/point3 objects)
                        if "point1" in ang and "point2" in ang and "point3" in ang:
                            p1 = ang.get("point1")
                            p2 = ang.get("point2")
                            p3 = ang.get("point3")
                        # Fall back to old coords string format
                        elif "coords" in ang:
                            coords = parse_coords(ang.get("coords", ""))
                            if len(coords) >= 3:
                                p1, p2, p3 = coords[0], coords[1], coords[2]
                        
                        if p1 and p2 and p3:
                            # Calculate angle if not provided
                            value = ang.get("value")
                            if value is None:
                                # Calculate angle from three points
                                v1 = {"x": p1["x"] - p2["x"], "y": p1["y"] - p2["y"]}
                                v2 = {"x": p3["x"] - p2["x"], "y": p3["y"] - p2["y"]}
                                dot = v1["x"] * v2["x"] + v1["y"] * v2["y"]
                                mag1 = (v1["x"]**2 + v1["y"]**2) ** 0.5
                                mag2 = (v2["x"]**2 + v2["y"]**2) ** 0.5
                                if mag1 > 0 and mag2 > 0:
                                    cos_val = max(-1, min(1, dot / (mag1 * mag2)))
                                    import math
                                    value = math.acos(cos_val) * 180 / math.pi
                                else:
                                    value = 0
                            frame_ann["measurements"]["angles"].append({
                                "id": ang.get("id", f"angle-{i+1}"),
                                "label": ang.get("name", ang.get("label", f"#{i+1}")),
                                "point1": p1,
                                "point2": p2,
                                "point3": p3,
                                "value": value,
                                "isReflex": ang.get("isReflex", False),
                                "color": ang.get("color", "#4fc3f7")
                            })
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {annotations_path}: {e}")
                    continue
        
        # Write to temp file
        temp_annotations = {study_id: {}}
        for frame_idx, frame_ann in all_frame_annotations.items():
            temp_annotations[study_id][str(frame_idx)] = frame_ann
        
        with temp_annotations_file.open("w", encoding="utf-8") as f:
            json.dump(temp_annotations, f, indent=2)
        
        # Also load into memory
        if study_id not in _frame_annotations:
            _frame_annotations[study_id] = {}
        for frame_idx, frame_ann in all_frame_annotations.items():
            _frame_annotations[study_id][frame_idx] = frame_ann
        
        logger.info(f"Loaded {len(all_frame_annotations)} frames with annotations from {annotations_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading study annotations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def list_studies() -> List[str]:
    """List all saved study IDs."""
    if not STUDY_DIR.exists():
        return []
    studies = []
    for d in STUDY_DIR.iterdir():
        if d.is_dir():
            # Check for JSONL (new format) or JSON (old format)
            if (d / f"{d.name}_annotations.jsonl").exists() or (d / f"{d.name}_annotations.json").exists():
                studies.append(d.name)
    return studies


@app.get("/study/{study_id}")
async def get_study(study_id: str) -> JSONResponse:
    """
    Load a saved study for viewing.
    This endpoint:
    1. Loads study metadata
    2. Finds and copies the video file to temp (with new video_id)
    3. Copies keypoints file to temp if it exists
    4. Parses JSONL annotations and writes to temp format
    5. Returns all necessary data for the frontend
    """
    study_dir = STUDY_DIR / study_id
    
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found")
    
    # Load metadata
    metadata = load_study(study_id)
    if not metadata:
        metadata = {}
    
    original_filename = metadata.get("original_filename", "")
    
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
    temp_video_path = TEMP_DIR / f"video_{video_id}{video_extension}"
    
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
    
    # Check for keypoints file in study folder and copy to temp
    keypoints_tracks = {}
    has_keypoints = False
    
    # Look for keypoints file in study folder (might be named differently)
    for f in study_dir.iterdir():
        if f.suffix.lower() == ".jsonl" and "keypoints" in f.name.lower():
            keypoints_path = TEMP_DIR / f"keypoints_{video_id}.jsonl"
            try:
                shutil.copy2(f, keypoints_path)
                keypoints_tracks = load_keypoints_tracks(keypoints_path)
                has_keypoints = len(keypoints_tracks) > 0
                logger.info(f"Copied keypoints file to temp: {keypoints_path}")
            except Exception as e:
                logger.warning(f"Failed to copy keypoints file: {e}")
            break
    
    # Load annotations from JSONL to temp
    load_study_annotations_to_temp(study_id)
    
    # Store in temp_studies for propagation support
    _temp_studies[video_id] = {
        "video_id": video_id,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": total_frames,
        "metadata": {
            "video_filename": original_filename,
            "total_frames": total_frames,
            "has_keypoints": has_keypoints,
        },
    }
    
    response_data = {
        "study_id": study_id,
        "video_id": video_id,
        "original_filename": original_filename,
        "frames": frames,
        "keypoints_tracks": keypoints_tracks,
        "total_frames": total_frames,
        "kind": "video",
        "metadata": {
            "video_filename": original_filename,
            "total_frames": total_frames,
            "has_keypoints": has_keypoints,
        },
    }
    
    return JSONResponse(response_data)


@app.get("/studies")
async def get_study_list() -> JSONResponse:
    return JSONResponse({"studies": list_studies()})


@app.get("/study/{study_id}/export")
async def export_study_jsonl(study_id: str) -> FileResponse:
    """
    Export study annotations - just returns the saved annotations.jsonl file.
    The saved file is already in the correct format (JSONL with name/action added).
    """
    study_dir = STUDY_DIR / study_id
    annotations_path = study_dir / f"{study_id}_annotations.jsonl"
    
    if not annotations_path.exists():
        raise HTTPException(status_code=404, detail="Study annotations file not found")
    
    return FileResponse(
        annotations_path,
        media_type="application/x-ndjson",
        filename=f"{study_id}_annotations.jsonl"
    )


@app.post("/study/{study_id}/propagate_labels")
async def propagate_labels(
    study_id: str,
    payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """
    Propagate labels forward only (from frame_index onwards) for a track ID.
    Does not overwrite existing labels in earlier frames.
    payload: {track_id, frame_index, name, action, video_id}
    """
    track_id = payload.get("track_id")
    frame_index = payload.get("frame_index")
    name = payload.get("name")
    action = payload.get("action")
    video_id = payload.get("video_id")  # Optional: for temp studies
    
    if track_id is None or frame_index is None:
        raise HTTPException(status_code=400, detail="track_id and frame_index are required")
    
    # Try to load study from disk first
    study_data = load_study(study_id)
    
    # If study not found and video_id provided, try to use temp study data
    if not study_data and video_id:
        temp_data = _temp_studies.get(video_id)
        if temp_data:
            study_data = temp_data
            logger.info(f"Using temp study data for video_id {video_id}")
    
    if not study_data:
        raise HTTPException(status_code=404, detail="Study not found. Please save the study first or provide video_id.")
    
    total_frames = study_data.get("total_frames", 0)
    keypoints_tracks = study_data.get("keypoints_tracks", {})
    
    # Propagate labels forward only (from frame_index onwards)
    updated_frames = []
    
    for frame_idx in range(frame_index, total_frames):
        # Check if this frame has the track_id
        frame_tracks = keypoints_tracks.get(frame_idx, [])
        has_track = any(track.get("track_id") == track_id for track in frame_tracks)
        
        if has_track:
            # Get or create annotations for this frame
            if study_id not in _frame_annotations:
                _frame_annotations[study_id] = {}
            if frame_idx not in _frame_annotations[study_id]:
                _frame_annotations[study_id][frame_idx] = {"bounding_boxes": {}}
            
            if "bounding_boxes" not in _frame_annotations[study_id][frame_idx]:
                _frame_annotations[study_id][frame_idx]["bounding_boxes"] = {}
            
            bbox_data = _frame_annotations[study_id][frame_idx]["bounding_boxes"].get(track_id, {})
            
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
            
            _frame_annotations[study_id][frame_idx]["bounding_boxes"][track_id] = bbox_data
            updated_frames.append(frame_idx)
            
            # Also save to server immediately
            annotations = {
                "keypoints": _frame_annotations[study_id][frame_idx].get("keypoints", []),
                "lines": _frame_annotations[study_id][frame_idx].get("lines", []),
                "rois": _frame_annotations[study_id][frame_idx].get("rois", []),
                "measurements": _frame_annotations[study_id][frame_idx].get("measurements", {"distances": [], "angles": []}),
                "bounding_boxes": _frame_annotations[study_id][frame_idx]["bounding_boxes"],
            }
            
            # Save this frame's annotations
            temp_annotations_file = TEMP_DIR / f"{study_id}_annotations.json"
            try:
                all_annotations = {}
                if temp_annotations_file.exists():
                    with temp_annotations_file.open("r", encoding="utf-8") as f:
                        all_annotations = json.load(f)
                
                if study_id not in all_annotations:
                    all_annotations[study_id] = {}
                all_annotations[study_id][str(frame_idx)] = annotations
                
                with temp_annotations_file.open("w", encoding="utf-8") as f:
                    json.dump(all_annotations, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save temp annotations file: {e}")
    
    return JSONResponse({
        "status": "propagated",
        "updated_frames": updated_frames,
        "count": len(updated_frames)
    })

@app.post("/reset_session")
async def reset_session(clear_temp: bool = True):
    """
    Clears all in-memory annotations and optionally temp files to ensure a fresh state.
    
    Args:
        clear_temp: If True, also clears temp folder files (default: True)
    """
    global _frame_annotations, _temp_studies
    
    # Clear in-memory annotations
    _frame_annotations.clear()
    _temp_studies.clear()
    
    # Clear video cache
    with _video_cache_lock:
        for video_id, cache_entry in _video_cache.items():
            try:
                cache_entry["capture"].release()
            except Exception:
                pass
        _video_cache.clear()
    
    files_cleared = 0
    
    # Clear temp folder files if requested
    if clear_temp and TEMP_DIR.exists():
        try:
            for item in TEMP_DIR.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        files_cleared += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        files_cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to delete temp item {item}: {e}")
            logger.info(f"Cleared temp directory: {files_cleared} items deleted")
        except Exception as e:
            logger.warning(f"Error clearing temp directory: {e}")
    
    return JSONResponse({"status": "session_cleared", "files_cleared": files_cleared})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
