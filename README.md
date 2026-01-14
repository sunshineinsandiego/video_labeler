# Video Labeler

A video annotation tool adapted from the MRI labeler for labeling video frames with bounding boxes, keypoints, and track annotations.

## Features

- Upload video files and keypoints_tracks.jsonl files
- Navigate through video frames with arrow keys or buttons
- View detected bounding boxes and keypoints on each frame
- Select bounding boxes and label them with player names and actions
- Propagate labels forward/backward through all frames with the same track ID
- Add annotations (keypoints, lines, ROIs, distances, angles) associated with bounding boxes
- Save and load studies with all annotations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python server.py
```

2. Open your browser to `http://localhost:8000`

3. Upload a video file and optionally a keypoints_tracks.jsonl file

4. Navigate frames using:
   - Left/Right arrow keys
   - Previous/Next frame buttons
   - Click on bounding boxes to select them

5. Label bounding boxes:
   - Select a bounding box by clicking on it
   - Enter player name and/or action
   - Click "Apply Labels" to label current frame
   - Click "Propagate to All Frames" to label all frames with the same track ID

6. Add annotations:
   - Use the annotation tools (keypoints, lines, ROIs, distances, angles)
   - These annotations are associated with the selected bounding box

7. Save your work:
   - Enter a Study ID
   - Click "Save study"

## Keypoints Tracks Format

The keypoints_tracks.jsonl file should have one JSON object per line with the following format:

```json
{
  "frame": 0,
  "track_id": 1,
  "bbox": [x1, y1, x2, y2],
  "keypoints": [x1, y1, conf1, x2, y2, conf2, ...],
  "score": 0.99,
  ...
}
```

- `frame`: Frame index (0-based)
- `track_id`: Unique track identifier
- `bbox`: Bounding box as [x1, y1, x2, y2]
- `keypoints`: Array of [x, y, confidence] triplets (17 keypoints = 51 values)

## Directory Structure

```
video_labeler/
├── server.py          # FastAPI backend
├── static/
│   ├── index.html     # Frontend HTML
│   ├── script.js      # Frontend JavaScript
│   └── styles.css     # Frontend CSS
├── data/
│   ├── uploads/       # Temporary uploads
│   ├── temp/          # Temporary files
│   └── studies/       # Saved studies
└── requirements.txt   # Python dependencies
```
