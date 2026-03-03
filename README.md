# Video Annotate

Video annotation tool for track labeling plus manual annotations (keypoints, lines, distances, angles, ROIs), with login, per-user storage, and save/load workflows.

## Current Login Credentials

Use these credentials for this repo state:

- Admin:
  - Email: `cd2859@cumc.columbia.edu`
  - Password: `admin123`
- User:
  - Email: `christopher.kip.robin@gmail.com`
  - Password: `user123`

## Supported Tracks Inputs

Optional tracks upload supports:

- `.jsonl` (existing format)
- `.pt` (Torch format)

`.pt` mapping used by this app:

- `track_id = id`
- `frame = frame_idx`
- `bbox = [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]`
- `keypoints` from `pred_2d` with confidence `1.0` for all 27 points
- For `.pt` uploads, keypoints are not rendered in the UI (tracks and bounding boxes still render).

## Runbook (Docker Compose Only)

1. Clone and enter the repo:
   ```bash
   git clone <YOUR_REPO_URL> video_labeler
   cd video_labeler
   ```

2. Build and run with Docker Compose:
   - Default port:
     ```bash
     VIDEO_LABELER_PORT=8000 docker compose up -d --build
     ```
   - Port `8002`:
     ```bash
     VIDEO_LABELER_PORT=8002 docker compose up -d --build
     ```

3. Set/reset the known passwords in container:
   ```bash
   docker compose exec video_labeler python manage_users.py reset-password --email cd2859@cumc.columbia.edu --password admin123
   docker compose exec video_labeler python manage_users.py reset-password --email christopher.kip.robin@gmail.com --password user123
   ```
   If either user does not exist, create it:
   ```bash
   docker compose exec video_labeler python manage_users.py create-admin --email cd2859@cumc.columbia.edu --password admin123
   docker compose exec video_labeler python manage_users.py create-user --email christopher.kip.robin@gmail.com --password user123
   ```

4. Access:
   - `http://localhost:<VIDEO_LABELER_PORT>`
   - `http://<VM_IP>:<VIDEO_LABELER_PORT>`

5. Logs:
   ```bash
   docker compose logs -f
   ```

6. If remote access is blocked, open firewall port:
   ```bash
   sudo ufw allow 8002/tcp
   ```

## Upload / Save / Load Behavior

- Upload:
  - Required: video (`.mp4`, `.avi`, `.mov`, `.mkv`)
  - Optional: tracks (`.jsonl` or `.pt`)
- Save:
  - New upload + new Study ID: create new study folder
  - New upload + existing Study ID: overwrite study folder (video + tracks + annotations)
  - Loaded study + same Study ID: update annotations only
  - Loaded study + new Study ID (Save As): copy video/tracks into new study and write annotations

## Data Layout

```
data/
├── users.db
├── studies/
│   └── <user_key>/
│       └── <study_id>/
│           ├── <video_filename>
│           ├── keypoints_tracks.jsonl
│           ├── <study_id>_annotations.jsonl
│           └── <study_id>_metadata.json
└── temp/
    └── <user_key>/
        └── ...
```

Notes:

- `data/` is persistent app state. Back it up.
- Temp working files are cleared on server startup.
- Passwords are stored hashed in `data/users.db`.

## Troubleshooting

- Container won't start:
  - Run `docker compose logs -f` and inspect startup errors.
- Login fails:
  - Re-run reset commands above inside container.
- No tracks rendered:
  - Confirm optional tracks file was uploaded and in supported format (`.jsonl` or `.pt`).
  - Check server logs for parse errors.
- Port already in use on host:
  ```bash
  lsof -i :8002
  ```

## Additional Docs

- Docker deployment details: `DOCKER.md`
- End-user UI guide: open `/README.html` in the app

## TODO

- Validate `.pt` file functionality end-to-end (upload, render, label, save, reload).
