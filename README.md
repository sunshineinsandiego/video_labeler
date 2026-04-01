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

## Deployment and Runbook (Docker)

### Prerequisites

- Docker installed on your system
- Docker Compose (recommended for easier deployment)

### 1. Clone and enter the repo:

```bash
git clone <YOUR_REPO_URL> video_labeler
cd video_labeler
```

### 2. Run with Docker Compose:

You can optionally customize the deployment using environment variables (e.g., `VIDEO_LABELER_PORT`, default: `8002`).

To build the image (required for the first run or after code changes), include the `--build` flag:

- Run on default port (`8002`):
  ```bash
  VIDEO_LABELER_PORT=8002 docker compose up -d --build
  ```
- Run on a different port (e.g., `8000`):
  ```bash
  VIDEO_LABELER_PORT=8000 docker compose up -d --build
  ```

**To run without rebuilding:** If the image is already built and you haven't made any code changes, you can start the container much faster by omitting the `--build` flag:
```bash
VIDEO_LABELER_PORT=8002 docker compose up -d
```

### 3. Set/reset the known passwords in container:

To set or change user passwords, run the following commands inside the container:

```bash
docker compose exec video_labeler python manage_users.py reset-password --email cd2859@cumc.columbia.edu --password admin123
docker compose exec video_labeler python manage_users.py reset-password --email christopher.kip.robin@gmail.com --password user123
```

If either user does not exist, create it:

```bash
docker compose exec video_labeler python manage_users.py create-admin --email cd2859@cumc.columbia.edu --password admin123
docker compose exec video_labeler python manage_users.py create-user --email christopher.kip.robin@gmail.com --password user123
```

### 4. Accessing the Web Interface:

**If running locally:**
- Open your browser and navigate to `http://localhost:<VIDEO_LABELER_PORT>` (default `8000`)

**If running remotely (Headless Server):**
To securely access the web interface from your local machine, use SSH local port forwarding. For example, if your app runs on port 8000 remotely:

```bash
ssh -i ~/.ssh/aws.pem -L 8002:localhost:8002 ubuntu@172.25.16.63
```

Then open your local browser and navigate to `http://localhost:8000`.

*Alternative (Direct Access):* If remote access is allowed directly, open the port in the firewall and access via the VM's IP address:
```bash
sudo ufw allow 8000/tcp
```
Then navigate to `http://<VM_IP>:8000`

### 5. Managing the Container:

- **View logs:**
  ```bash
  docker compose logs -f
  ```
- **Stop the container:**
  ```bash
  docker compose down
  ```

### 6. Updating the Application:

1. Pull latest changes:
   ```bash
   git pull
   ```
2. Rebuild and restart:
   ```bash
   docker compose up -d --build
   ```
*Note: Your data directory will persist across updates.*

## Upload / Save / Load Behavior

- Upload:
  - Required: video (`.mp4`, `.avi`, `.mov`, `.mkv`)
  - Optional: tracks (`.jsonl` or `.pt`)
- Save:
  - New upload + new Study ID: create new study folder
  - New upload + existing Study ID: overwrite study folder (video + tracks + annotations)
  - Loaded study + same Study ID: update annotations only
  - Loaded study + new Study ID (Save As): copy video/tracks into new study and write annotations

## Data Layout and Persistence

The `data/` directory is bind-mounted into the container, meaning all studies, uploads, and annotations are safely persisted on your host filesystem.

```
data/
├── users.db
├── temp/
│   └── <user_key>/
│       └── ...
└── studies/
    └── <user_key>/
        └── <study_id>/
            ├── <video_filename>
            ├── keypoints_tracks.jsonl
            ├── <study_id>_annotations.jsonl
            └── <study_id>_metadata.json
```

Notes:
- `data/` is the persistent app state. It remains intact even if the container is removed.
- Temp working files are cleared on server startup.
- Passwords are stored hashed in `data/users.db`.

**Backing up the data:**
```bash
# Create a backup
tar czf video_labeler_data_backup.tar.gz data/

# Restore from backup
tar xzf video_labeler_data_backup.tar.gz
```

## Troubleshooting

- **Container won't start:**
  - Run `docker compose logs -f` and inspect startup errors.
  - Verify port is available: `netstat -tuln | grep 8000`
- **Login fails:**
  - Re-run the `reset-password` commands above inside the container.
- **No tracks rendered:**
  - Confirm optional tracks file was uploaded and in supported format (`.jsonl` or `.pt`).
  - Check server logs for parse errors.
- **Port already in use on host:**
  - Find what is using it: `lsof -i :8000` (or `8002`)
- **Can't access from remote machine:**
  - Check firewall: `sudo ufw status`
  - Verify container is running: `docker ps`
  - Check if port is bound correctly: `docker port video_labeler`
- **Data not persisting:**
  - Check data directory exists: `ls -la data/`
  - Verify bind mount: `docker inspect video_labeler | grep -A 10 Mounts`

## Additional Docs

- End-user UI guide: open `/README.html` in the app