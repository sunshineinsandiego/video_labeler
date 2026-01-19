# Video Annotate

A video annotation tool for tracking-based labeling plus manual annotations, with per-user storage and save/load workflows.

## Features

- Login-required access (users stored in SQLite)
- Upload a video file and optional `keypoints_tracks.jsonl`
- Frame navigation via buttons and arrow keys
- Track labeling (player name/action) with forward propagation
- Manual annotations: keypoints, lines, distances, angles, ROIs
- Save, load, overwrite, and save-as workflows

## Quick Start (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create users:
   ```bash
   python manage_users.py create-admin --email you@example.com --password <password>
   python manage_users.py create-user --email user@example.com --password <password>
   ```

3. Start the server:
   ```bash
   python server.py
   ```

4. Open `http://localhost:8000` and log in.

## Save / Load Behavior

- **New upload + new Study ID**: creates a new study folder.
- **New upload + existing Study ID**: overwrites the existing study folder (video + tracks + annotations).
- **Load + save same Study ID**: updates annotations only; video and keypoints tracks remain unchanged.
- **Load + save new Study ID (Save As)**: copies the video and keypoints file into a new study folder and writes new annotations.

## Keypoints Tracks Format

`keypoints_tracks.jsonl` should contain one JSON object per line, for example:

```json
{
  "frame": 0,
  "track_id": 1,
  "bbox": [x1, y1, x2, y2],
  "keypoints": [x1, y1, conf1, x2, y2, conf2, ...],
  "score": 0.99
}
```

## Data Layout

```
data/
├── users.db
├── studies/
│   └── <user_key>/
│       └── <study_id>/
│           ├── <video_filename>
│           ├── keypoints_tracks.jsonl
│           └── <study_id>_annotations.jsonl
└── temp/
    └── <user_key>/
        └── ...                         # Current working session
```

## Docker

See `DOCKER.md` for container deployment.

## Help Guide

Open `README.html` in a browser for the end-user guide.
