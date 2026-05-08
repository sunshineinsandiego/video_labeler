# Video Annotate

Video annotation tool for **track labels** (name / action per bounding box) plus **manual geometry** (keypoints, lines, distances, angles, ROIs). Uses **login**, **per-user storage**, and **save / load** workflows.

**Tracks I/O:** exactly **two uploads** per session start — **video** and a **CoMotion-style MOT `.txt`**. The app does **not** read `.pt`, `.jsonl`, or separate JSON **annotation** files (geometry lives as a JSON **string** inside each extended MOT `.txt` line; see [Extended MOT](#extended-mot-study_id_annotationstxt)).

**Studies are private per login:** each account has its own directory tree under `data/studies/<user_key>/` and `data/temp/<user_key>/`. The same **study ID** string used by two different logins refers to **two different folders**. Share work between accounts by copying the study folder or using **Export** (`GET /study/{study_id}/export`), not by “using the same ID.”

---

## Current login credentials (this repo)

- **Admin:** `cd2859@cumc.columbia.edu` / `admin123`
- **User:** `christopher.kip.robin@gmail.com` / `user123`

---

## Quick start (operator)

1. **Run** the app (Docker — see [Deployment](#deployment-and-runbook-docker)).
2. **Log in** in the browser.
3. **Select video** and **MOT tracks `.txt`** (both required). Click **Upload**.
4. **Pick a bbox** on the canvas (click inside a box). Set **Player name** / **Action** and **Apply labels** (labels can propagate forward by track).
5. Use **Create keypoint / line / ROI / distance / angle** while a bbox is selected; geometry is stored **per selected track row** for the current frame in the extended MOT file.
6. Enter a **Study ID** and **Save study**.

In-app help: open **`/README.html`** after logging in.

---

## Inputs (what you must provide)

| Input | Required | Accepted | Notes |
|--------|----------|----------|--------|
| Video | Yes | `.mp4`, `.avi`, `.mov`, `.mkv` | Shown frame-by-frame via OpenCV; UI frame index is **0-based**. |
| Tracks | Yes | `.txt` | **Comma-separated MOT** with **exactly 10 fields** per row, as produced by [apple/ml-comotion](https://github.com/apple/ml-comotion) `convert_to_mot` (see [MOT .txt format](#mot-txt-format-comotion)). |

**Not supported in-app:** `.pt` checkpoints, `keypoints_tracks.jsonl`, or any separate “keypoints file” upload. CoMotion may still emit a `.pt` for research; you only need the **`stem.txt`** (plus video) here.

**HTTP upload:** `POST /upload` multipart form fields:

- `video_file` — video
- `tracks_txt_file` — MOT text

---

## Web UI overview

After upload, the main view includes:

- **Video + tracks file** pickers and **Upload** (clears your server-side **temp** workspace for that login and loads the new pair).
- **Study ID** field plus **Save study** and **Load study**.
- **Canvas** — current frame; **click a bbox** to select a **track** (`track_id`). Labels and manual tools apply to the **selected** track.
- **Frame navigation** — prev/next, jump by frame number; keyboard **Left** / **Right**.
- **Player name / Action** — per-track, per-frame (and propagation — see **`POST .../propagate_labels`**).
- **Manual tools** — keypoints, lines, ROIs, distances, angles; edits are persisted in the extended MOT **per MOT line** (per detection / track on that frame).

**Multi-tab / restart:** Loading a study **clears all temp files** for that login. After a **server restart**, **all of `data/temp`** is wiped at process startup; reload the page and **upload or load study** again. If one tab loads a study while another tab still holds an old `video_id`, annotation requests may return **404** until you refresh.

---

## MOT .txt format (CoMotion)

Reference: CoMotion `convert_to_mot` / `track.py` — [ml-comotion](https://github.com/apple/ml-comotion).

Per line (comma-separated):

`frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z`

- **Exactly 10 fields** per non-empty row (required). Malformed rows **fail** on upload, study load, and on any path that parses the extended MOT file (no silent skipping).
- **Column 0:** MOT frame index — **1-based** (`frame_idx + 1` in CoMotion’s 0-based timeline).
- **Column 1:** **track id** (integer).
- **Columns 2–5:** **left, top, width, height** as written by CoMotion (their pipeline applies **`+1`** to each of the four when serializing).
- **Columns 6–9:** commonly `1,0,0,0` (confidence / 3D placeholders).

**UI / OpenCV frame index:** `frame_ui = mot_frame - 1`.

**Bbox decoder:** subtract 1 from L, T, W, H, then  
`x1,y1,x2,y2 = L0, T0, L0+W0, T0+H0`.

A sample tracks file with 10-field rows lives at `test_data/0_4999.txt` (no sample video is committed here because of `.gitignore`; use your own `.mp4` with that tracks file, or any CoMotion export pair).

---

## Outputs (what gets saved)

### Study folder layout

Persisted under `data/studies/<user_key>/<study_id>/`:

| File | Role |
|------|------|
| `<video_filename>` | Copy of the uploaded / saved video (original name from metadata). |
| `<tracks_txt_filename>` | **Full safe basename** of the MOT tracks file from metadata (e.g. `clip.txt` or `0_4999.txt`). The code does **not** append an extra `.txt` beyond what is already in that stored name. |
| `<study_id>_annotations.txt` | **Extended MOT** — see below. |
| `<study_id>_metadata.json` | JSON sidecar: `video_filename`, `tracks_txt_filename`, `tracks_source_format` (`mot_txt`), `study_id`, `video_id` (session id **at last save** — **not** a substitute for the current temp `video_id` in the API; see API notes). |

**Listing studies:** `GET /studies` returns study IDs that have **`<study_id>_annotations.txt`** (per logged-in user only).

### Extended MOT: `<study_id>_annotations.txt`

- **One output line per input MOT line**, **same order** as the source tracks `.txt`.
- Each line = **exact same comma-separated MOT prefix** as the source row (still **10 fields**, 1-based frame in column 0). **No separate JSON annotation file:** the fourth tab field is a **JSON string** (keypoints, lines, rois, measurements) for tools on **that** detection row.
- Then **three tab-separated fields:** `name`, `action`, `manual` (JSON string as above).
- Sync from the UI updates **per row** matching `(mot_frame, track_id)`; the **selected** track can receive the current canvas payload’s global manual tools into that row’s JSON when appropriate ([`mot_tracks.sync_payload_to_extended_mot_file`](mot_tracks.py)).

**Example** (tabs as `\t`):

```text
1,3,12,22,32,42,1,0,0,0\tAlice\trun\t{"keypoints":[],"lines":[],"rois":[],"measurements":{"distances":[],"angles":[]}}
```

**GET JSON note:** responses may include **top-level** `keypoints` / `lines` / `rois` / `measurements` copied from the **first** non-empty per-row manual on that frame (convenience for the UI). **Canonical** per-track geometry is `bounding_boxes[track_id].annotations`, aligned with the row’s tab JSON.

### Export

- `GET /study/{study_id}/export` — downloads `<study_id>_annotations.txt` as `text/plain`.

### Session (temporary) files

Under `data/temp/<user_key>/`:

- On **every application process start**, the server **clears the entire `data/temp` tree** (all users’ temp files).
- During a session: `video_<uuid>.<ext>`, `tracks_<uuid>.txt`, `{study_id}_annotations.txt` (working extended MOT). **Save study** copies the extended MOT (and tracks/video) into the study folder.

There is **no** `{study_id}_annotations.json` temp store for annotations.

---

## Save / load behavior (summary)

- **Upload:** replaces **temp** workspace for the user; requires **video + MOT `.txt`**.
- **Save study (`POST /study/{study_id}/save`):** writes video + tracks + flushes in-memory frame edits into temp extended MOT (merge honors **empty lists** from the client so cleared tool geometry persists), then copies **`{study_id}_annotations.txt`** and MOT tracks into the study directory; writes metadata.
- **Load study (`GET /study/{study_id}`):** **clears** that user’s temp and in-memory caches, then copies persisted video + tracks + annotations into a **new** temp `video_id`, returns frames and `keypoints_tracks` for the UI.

---

## HTTP API (subset)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload` | `video_file`, `tracks_txt_file` — returns `video_id`, `frames`, `keypoints_tracks`, `metadata`. |
| GET | `/video/{video_id}/frame/{frame_index}` | JPEG frame image. |
| GET | `/video/{video_id}/keypoints/{frame_index}` | JSON `{ "tracks": [...] }` for that **0-based** frame. |
| GET | `/study/{study_id}/frame/{frame_index}/annotations` | **Requires query `video_id`** (current temp session id). Requires **`{study_id}_annotations.txt` in temp** and **`tracks_{video_id}.txt` in temp**. No fallback to study-disk copy or metadata `video_id`. **404/400** if missing or invalid MOT. |
| POST | `/study/{study_id}/frame/{frame_index}/annotations` | Body JSON; updates temp extended MOT when `video_id` in body is set. |
| POST | `/study/{study_id}/save` | Form `payload` JSON + persists study assets. |
| GET | `/study/{study_id}` | Load saved study into temp (clears prior temp for this user). |
| GET | `/studies` | List study IDs for this login (must have `_annotations.txt`). |
| GET | `/study/{study_id}/export` | Download extended MOT annotations `.txt`. |
| POST | `/study/{study_id}/propagate_labels` | Forward label propagation for a track (bounding box keys normalized to strings server-side). |

---

## Plan vs code (consistency checklist)

| Plan item | Status |
|-----------|--------|
| Two-file upload (video + `.txt`), hard-fail if tracks missing | Implemented: `POST /upload`, UI requires both files. |
| MOT rows: **exactly 10 fields**, hard-fail everywhere | Implemented: [`mot_tracks.parse_mot_bbox_only`](mot_tracks.py), extended MOT read/sync. |
| No `.pt` / no torch for tracks | Implemented. |
| No JSONL in app paths | Implemented. |
| No temp JSON for annotations | Implemented: temp uses `{study_id}_annotations.txt` only. |
| Extended MOT rows: MOT prefix + `\tname\taction\tmanual` (manual = JSON string) | Implemented. |
| Same line order as source MOT | Implemented: `ensure_extended_mot_copy`. |
| `list_studies` requires `_annotations.txt` | Implemented. |
| Studies **per login** (`data/studies/<user_key>/`) | Implemented: `_user_dirs` in [server.py](server.py). |
| `GET .../annotations`: **required `video_id`**, temp-only, no stale-metadata fallback | Implemented. |
| Merge flush: **empty lists** from client overwrite disk fields | Implemented: `_merge_frame_payload_disk_and_memory`. |

---

## Behavior and API notes (recent changes)

- **Strict MOT:** Every non-empty data row must have **10 comma-separated fields**; invalid rows cause **400** on upload, save flush, propagate sync, and strict GET parsing — not silent skips.
- **Strict `GET .../annotations`:** Query parameter **`video_id` is required** (must match temp `tracks_{video_id}.txt`). Temp **`{study_id}_annotations.txt` must exist** (typically after the first `POST` for that study or after **load study**). The server does **not** fall back to reading the study folder copy or inventing lines from tracks for this GET.
- **Merge / save:** Clearing keypoints, lines, or rois in the UI (empty arrays in the POST body) is merged with **key presence** so **empties persist** to the extended MOT on save/flush.
- **Propagated labels:** In-memory `bounding_boxes` keys are normalized to **strings** to avoid `3` vs `"3"` duplicates.

---

## Deployment and runbook (Docker)

### Prerequisites

- Docker and Docker Compose.

### Clone and run

```bash
git clone <YOUR_REPO_URL> video_labeler
cd video_labeler
```

Default host port is **`8000`** unless you override `VIDEO_LABELER_PORT`:

```bash
docker compose up -d --build
# or
VIDEO_LABELER_PORT=8002 docker compose up -d --build
```

Open: **`http://localhost:<VIDEO_LABELER_PORT>`** (default **`8000`**).

### User management (inside container)

Run every [`manage_users.py`](manage_users.py) command with **`docker compose exec`** against the running **`video_labeler`** service (same DB file and dependencies as the app):

```bash
docker compose exec video_labeler python manage_users.py <subcommand> [options]
```

Credentials live in **`data/users.db`** on the host (bind-mounted into the container).
After pulling code changes to `manage_users.py`, rebuild before running `docker compose exec` so new subcommands are available: `docker compose up -d --build`.

**Create an admin** (default email is the value of `DEFAULT_ADMIN_EMAIL` in `manage_users.py` unless you pass `--email`):

```bash
docker compose exec video_labeler python manage_users.py create-admin --password YOUR_ADMIN_PASSWORD
docker compose exec video_labeler python manage_users.py create-admin --email admin@example.com --password YOUR_ADMIN_PASSWORD
```

**Create a regular user:**

```bash
docker compose exec video_labeler python manage_users.py create-user --email user@example.com --password YOUR_PASSWORD
```

**List all users:**

```bash
docker compose exec video_labeler python manage_users.py list
```

**Delete a user** (SQLite row only; files under `data/studies/<user_key>/` are **not** removed):

```bash
docker compose exec video_labeler python manage_users.py delete-user --email user@example.com
```

**Reset a password:**

```bash
docker compose exec video_labeler python manage_users.py reset-password --email user@example.com --password NEW_PASSWORD
```

### One-off Python in the app image

```bash
docker compose run --rm video_labeler python3 -c "import server, mot_tracks; print('ok')"
```

### Logs and lifecycle

```bash
docker compose logs -f
docker compose down
```

After pulling new code: `docker compose up -d --build`. The `data/` bind mount persists **studies**; **temp** is wiped on every app **startup**.

**Scaling note:** Annotation file locking uses in-process `threading.Lock` only; multiple Uvicorn workers on a shared `data/` volume can race on the same files. Prefer a **single worker** unless you add external locking or atomic file replace semantics.

---

## Data layout and persistence

`data/` is bind-mounted (see `docker-compose.yml`):

```
data/
├── users.db
├── temp/                          # cleared entirely on each server process start
│   └── <user_key>/
│       ├── video_<uuid>.<ext>
│       ├── tracks_<uuid>.txt
│       └── <study_id>_annotations.txt   # session extended MOT
└── studies/
    └── <user_key>/                 # private to that login
        └── <study_id>/
            ├── <video_filename>
            ├── <tracks_txt_filename>       # MOT source copy (basename from metadata)
            ├── <study_id>_annotations.txt  # extended MOT output
            └── <study_id>_metadata.json
```

**Backup:**

```bash
tar czf video_labeler_data_backup.tar.gz data/
tar xzf video_labeler_data_backup.tar.gz
```

---

## Troubleshooting

- **Container won’t start:** `docker compose logs -f`; check host port: `ss -tuln | grep 8000` (or your `VIDEO_LABELER_PORT`).
- **Login fails:** `docker compose exec video_labeler python manage_users.py reset-password --email <email> --password <new_password>`
- **Upload 400 / “Invalid MOT”:** every non-empty row needs **exactly 10 comma-separated fields** and valid numeric frame, id, and bbox columns; check server logs for the line number.
- **404 on GET annotations:** ensure you **uploaded or loaded study**, POST has run at least once for that temp `study_id`, and the client sends the current **`video_id`** query param. After **restart**, temp is empty — **reload study or re-upload**.
- **No tracks drawn:** confirm MOT frames overlap your video length; confirm upload succeeded.
- **Studies missing in Load list:** only folders with `<study_id>_annotations.txt` appear (for **your** login); legacy jsonl-only studies need migration (re-save or recreate).

---

## Additional docs

- End-user UI guide in the app: **`/README.html`**.
