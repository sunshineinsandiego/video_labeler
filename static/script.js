// Video Labeler - Adapted from MRI Labeler
const canvas = document.getElementById("image-canvas");
const ctx = canvas.getContext("2d");
const videoInput = document.getElementById("video-input");
const keypointsInput = document.getElementById("keypoints-input");
const uploadBtn = document.getElementById("upload-btn");
const videoFileInfo = document.getElementById("video-file-info");
const keypointsFileInfo = document.getElementById("keypoints-file-info");
const metadataOutput = document.getElementById("metadata-output");
const keypointList = document.getElementById("keypoint-list");
const lineList = document.getElementById("line-list");
const roiList = document.getElementById("roi-list");
const distanceList = document.getElementById("distance-list");
const angleList = document.getElementById("angle-list");
const studyIdInput = document.getElementById("study-id");
const bboxInfo = document.getElementById("bbox-info");
const playerNameInput = document.getElementById("player-name-input");
const actionInput = document.getElementById("action-input");
const applyLabelsBtn = document.getElementById("apply-labels-btn");

const createKeypointBtn = document.getElementById("create-keypoint-btn");
const drawLineBtn = document.getElementById("draw-line-btn");
const angleBtn = document.getElementById("angle-btn");
const distanceBtn = document.getElementById("distance-btn");
const roiBtn = document.getElementById("roi-btn");
const saveStudyBtn = document.getElementById("save-study-btn");
const loadStudyBtn = document.getElementById("load-study-btn");
const studyModal = document.getElementById("study-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const studyListContainer = document.getElementById("study-list-container");
const zoomInBtn = document.getElementById("zoom-in-btn");
const zoomOutBtn = document.getElementById("zoom-out-btn");
const resetViewBtn = document.getElementById("reset-view-btn");
const panUpBtn = document.getElementById("pan-up-btn");
const panDownBtn = document.getElementById("pan-down-btn");
const panLeftBtn = document.getElementById("pan-left-btn");
const panRightBtn = document.getElementById("pan-right-btn");
const frameNavigationInline = document.getElementById("frame-navigation-inline");
const prevFrameBtn = document.getElementById("prev-frame-btn");
const nextFrameBtn = document.getElementById("next-frame-btn");
const frameInfo = document.getElementById("frame-info");
const frameNumberDisplay = document.getElementById("frame-number-display");
const frameTotalDisplay = document.getElementById("frame-total-display");

const KEYPOINT_SIZE = 6;
const COLOR_OPTIONS = [
  "#ff5252", "#ff7043", "#ffab40", "#ffd740", "#eeff41",
  "#c6ff00", "#76ff03", "#64dd17", "#4caf50", "#00e676",
  "#00bcd4", "#00acc1", "#0097a7", "#00838f", "#006064",
  "#2196f3", "#1976d2", "#1565c0", "#0d47a1", "#0277bd",
  "#673ab7", "#5e35b1", "#512da8", "#4527a0", "#311b92",
];

// COCO-17 skeleton topology (0-based indices)
const COCO17_EDGES = [
  [0,1],[0,2],[1,3],[2,4],      // face
  [5,6],                        // shoulders
  [5,7],[7,9],                  // left arm
  [6,8],[8,10],                 // right arm
  [5,11],[6,12],[11,12],        // torso / hips
  [11,13],[13,15],              // left leg
  [12,14],[14,16],              // right leg
];

let colorIndex = 0;
let lineColorIndex = 0;
let distanceColorIndex = 0;
let angleColorIndex = 0;
let roiColorIndex = 0;

// Hash function to generate consistent colors per track_id (similar to viz.py)
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

function colorForTrackId(trackId) {
  // Generate a stable hue from track_id (0-179 range, matching OpenCV HSV)
  const hash = hashString(`track:${trackId}`);
  const hue = hash % 180;
  
  // Convert HSV to RGB (using vivid colors like viz.py: s=230, v=255)
  const h = hue / 180.0; // Normalize to 0-1
  const s = 230 / 255; // saturation
  const v = 255 / 255; // value
  
  const c = v * s;
  const x = c * (1 - Math.abs((h * 6) % 2 - 1));
  const m = v - c;
  
  let r, g, b;
  const h6 = h * 6;
  if (h6 < 1) {
    r = c; g = x; b = 0;
  } else if (h6 < 2) {
    r = x; g = c; b = 0;
  } else if (h6 < 3) {
    r = 0; g = c; b = x;
  } else if (h6 < 4) {
    r = 0; g = x; b = c;
  } else if (h6 < 5) {
    r = x; g = 0; b = c;
  } else {
    r = c; g = 0; b = x;
  }
  
  r = Math.round((r + m) * 255);
  g = Math.round((g + m) * 255);
  b = Math.round((b + m) * 255);
  
  return `rgb(${r}, ${g}, ${b})`;
}

const state = {
  image: null,
  imageUrl: null,
  videoId: null,
  storedFilename: null,
  originalFilename: null,
  kind: null,
  metadata: {},
  keypoints: [],
  lines: [],
  rois: [],
  measurements: { distances: [], angles: [] },
  mode: "idle",
  tempPoints: [],
  zoom: 1.0,
  panX: 0,
  panY: 0,
  dragging: null,
  draggingLineEndpoint: null,
  draggingDistanceEndpoint: null,
  draggingAnglePoint: null,
  draggingRoiPoint: null,
  dragStartX: 0,
  dragStartY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  previewMouseX: null,
  previewMouseY: null,
  nextKeypointIndex: 1,
  nextLineIndex: 1,
  nextRoiIndex: 1,
  nextDistanceIndex: 1,
  nextAngleIndex: 1,
  // Video-specific state
  frames: [],
  currentFrameIndex: 0,
  keypointsTracks: {}, // {frame_index: [tracks]}
  hasKeypointsFile: false, // Track if keypoints file was uploaded
  boundingBoxes: {}, // {track_id: {name, action, annotations}} - labels stored per track_id, applied to all frames
  selectedTrackId: null,
  studyId: null,
  tempStudyId: null,
  isNavigating: false, // Lock to prevent race conditions during frame navigation
};

function setMode(mode, message) {
  state.mode = mode;
  const buttons = {
    "add-keypoint": createKeypointBtn,
    "draw-line": drawLineBtn,
    "roi": roiBtn,
    "distance": distanceBtn,
    "angle": angleBtn,
  };
  Object.values(buttons).forEach(btn => {
    if (btn) btn.classList.remove("active");
  });
  const activeBtn = buttons[mode];
  if (activeBtn) {
    activeBtn.classList.add("active");
  }
}

function canvasToWorld(canvasX, canvasY) {
  const x = (canvasX - state.panX) / state.zoom;
  const y = (canvasY - state.panY) / state.zoom;
  return { x, y };
}

function worldToCanvas(worldX, worldY) {
  const x = worldX * state.zoom + state.panX;
  const y = worldY * state.zoom + state.panY;
  return { x, y };
}

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  resetView();
  render();
}

function resetView() {
  if (state.image) {
    const scaleX = canvas.width / state.image.width;
    const scaleY = canvas.height / state.image.height;
    state.zoom = Math.min(scaleX, scaleY) * 0.95;
    const scaledWidth = state.image.width * state.zoom;
    const scaledHeight = state.image.height * state.zoom;
    state.panX = (canvas.width - scaledWidth) / 2;
    state.panY = (canvas.height - scaledHeight) / 2;
  } else {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
  }
}

function zoomAtPoint(canvasX, canvasY, delta) {
  const worldX = (canvasX - state.panX) / state.zoom;
  const worldY = (canvasY - state.panY) / state.zoom;
  const zoomFactor = delta > 0 ? 1.1 : 0.9;
  const newZoom = Math.max(0.1, Math.min(10, state.zoom * zoomFactor));
  state.panX = canvasX - worldX * newZoom;
  state.panY = canvasY - worldY * newZoom;
  state.zoom = newZoom;
  render();
}

function panBy(dx, dy) {
  state.panX += dx;
  state.panY += dy;
  render();
}

window.addEventListener("resize", fitCanvas);

async function uploadVideo() {
  console.log("[uploadVideo] Function called");
  const videoFile = videoInput.files[0];
  const keypointsFile = keypointsInput.files[0];
  
  if (!videoFile) {
    alert("Please select a video file first.");
    return;
  }
  
  console.log(`[uploadVideo] Video file: ${videoFile.name}, size: ${videoFile.size}`);
  console.log(`[uploadVideo] Keypoints file: ${keypointsFile ? keypointsFile.name + ', size: ' + keypointsFile.size : 'none'}`);
  
  // Disable upload button during upload
  const uploadBtnText = document.getElementById("upload-btn-text");
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  
  if (uploadBtn) {
    uploadBtn.disabled = true;
    if (uploadBtnText) {
      uploadBtnText.textContent = "Uploading...";
    }
    if (uploadProgressBar) {
      uploadProgressBar.style.width = "0%";
    }
  }
  
  setMode("uploading", "Uploading video...");
  
  // Reset session and clear temp files to ensure fresh state
  try { 
    await fetch("/reset_session?clear_temp=true", { method: "POST" }); 
    console.log("[uploadVideo] Session reset and temp files cleared");
  } catch (e) {
    console.warn("[uploadVideo] Failed to reset session:", e);
  }
  
  // Reset client-side state
  state.studyId = null;
  state.tempStudyId = null;
  state.videoId = null;
  state.keypointsTracks = {};
  state.hasKeypointsFile = false;
  state.frames = [];
  state.currentFrameIndex = 0;
  state.boundingBoxes = {};
  state.selectedTrackId = null;
  resetAnnotations();
  
  const form = new FormData();
  form.append("video_file", videoFile);
  if (keypointsFile) {
    console.log(`[uploadVideo] Adding keypoints file to form: ${keypointsFile.name}`);
    form.append("keypoints_file", keypointsFile);
  } else {
    console.log(`[uploadVideo] No keypoints file selected - video will be uploaded without tracks`);
  }
  
  try {
    // Use XMLHttpRequest for progress tracking
    const xhr = new XMLHttpRequest();
    
    // Track upload progress
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && uploadProgressBar) {
        const percentComplete = (e.loaded / e.total) * 100;
        uploadProgressBar.style.width = `${percentComplete}%`;
        console.log(`[uploadVideo] Upload progress: ${percentComplete.toFixed(1)}%`);
      }
    });
    
    // Handle completion
    xhr.addEventListener("load", async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          await handleUploadResponse(data);
          
          // Reset progress bar
          if (uploadProgressBar) {
            uploadProgressBar.style.width = "100%";
            setTimeout(() => {
              uploadProgressBar.style.width = "0%";
            }, 300);
          }
          
          // Re-enable upload button
          if (uploadBtn) {
            uploadBtn.disabled = false;
            if (uploadBtnText) {
              uploadBtnText.textContent = "Upload";
            }
          }
          updateUploadButtonState();
        } catch (error) {
          console.error("Error parsing response:", error);
          alert(`Failed to parse server response: ${error.message}`);
          setMode("idle", "Upload failed");
          if (uploadBtn) {
            uploadBtn.disabled = false;
            if (uploadBtnText) {
              uploadBtnText.textContent = "Upload";
            }
            if (uploadProgressBar) {
              uploadProgressBar.style.width = "0%";
            }
          }
          updateUploadButtonState();
        }
      } else {
        const errorText = xhr.responseText || `HTTP ${xhr.status}`;
        alert(`Upload error: ${errorText}`);
        setMode("idle", "Upload failed");
        if (uploadBtn) {
          uploadBtn.disabled = false;
          if (uploadBtnText) {
            uploadBtnText.textContent = "Upload";
          }
          if (uploadProgressBar) {
            uploadProgressBar.style.width = "0%";
          }
        }
        updateUploadButtonState();
      }
    });
    
    // Handle errors
    xhr.addEventListener("error", () => {
      alert("Failed to upload video: Network error");
      setMode("idle", "Upload failed");
      if (uploadBtn) {
        uploadBtn.disabled = false;
        if (uploadBtnText) {
          uploadBtnText.textContent = "Upload";
        }
        if (uploadProgressBar) {
          uploadProgressBar.style.width = "0%";
        }
      }
      updateUploadButtonState();
    });
    
    // Start upload
    xhr.open("POST", "/upload");
    xhr.send(form);
    
  } catch (error) {
    console.error("Error uploading video:", error);
    alert(`Failed to upload video: ${error.message}`);
    setMode("idle", "Upload failed");
    if (uploadBtn) {
      uploadBtn.disabled = false;
      if (uploadBtnText) {
        uploadBtnText.textContent = "Upload";
      }
      if (uploadProgressBar) {
        uploadProgressBar.style.width = "0%";
      }
    }
    updateUploadButtonState();
  }
}

async function handleUploadResponse(data) {
  state.videoId = data.video_id;
  state.imageUrl = data.image_url;
  state.storedFilename = data.stored_filename;
  state.originalFilename = data.metadata.video_filename;
  state.kind = data.kind;
  state.metadata = data.metadata || {};
  state.frames = data.frames || [];
  
  // Normalize keypointsTracks keys to ensure both string and number keys work
  // JSON may serialize integer keys as strings, so we need to handle both
  const rawTracks = data.keypoints_tracks || {};
  state.keypointsTracks = {};
  state.hasKeypointsFile = data.metadata?.has_keypoints || false;
  
  console.log(`[handleUploadResponse] Raw keypoints_tracks type:`, typeof rawTracks, 'keys:', Object.keys(rawTracks).slice(0, 5));
  console.log(`[handleUploadResponse] Raw keypoints_tracks sample:`, rawTracks);
  console.log(`[handleUploadResponse] Has keypoints file:`, state.hasKeypointsFile);
  
  // Handle both object with integer keys and object with string keys
  if (rawTracks && typeof rawTracks === 'object' && Object.keys(rawTracks).length > 0) {
    state.hasKeypointsFile = true;
    for (const key in rawTracks) {
      if (rawTracks.hasOwnProperty(key)) {
        const numKey = parseInt(key, 10);
        const value = rawTracks[key];
        if (!isNaN(numKey) && Array.isArray(value)) {
          // Store with both number and string keys for compatibility
          state.keypointsTracks[numKey] = value;
          state.keypointsTracks[String(numKey)] = value;
        }
      }
    }
  }
  
  console.log(`[handleUploadResponse] Loaded keypointsTracks:`, Object.keys(state.keypointsTracks).length, 'frames');
  console.log(`[handleUploadResponse] Sample frame 0 tracks:`, state.keypointsTracks[0]?.length || 0, 'tracks');
  if (state.keypointsTracks[0] && state.keypointsTracks[0].length > 0) {
    console.log(`[handleUploadResponse] First track sample:`, state.keypointsTracks[0][0]);
  }
  
  state.currentFrameIndex = 0;
  state.boundingBoxes = {};
  state.selectedTrackId = null;
  
  state.keypoints = [];
  state.lines = [];
  state.rois = [];
  state.measurements = { distances: [], angles: [] };
  state.nextKeypointIndex = 1;
  state.nextLineIndex = 1;
  state.nextRoiIndex = 1;
  state.nextDistanceIndex = 1;
  state.nextAngleIndex = 1;
  
  if (!state.tempStudyId) {
    state.tempStudyId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  metadataOutput.textContent = JSON.stringify(state.metadata, null, 2);
  
  if (frameNavigationInline) frameNavigationInline.style.display = "inline-flex";
  
  // Setup frame number editor
  setupFrameNumberEditor();
  
  await loadFrame(0);
  setMode("idle", "Video loaded. Navigate frames with arrow keys.");
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      state.image = img;
      fitCanvas();
      // Render after a short delay to ensure canvas is ready
      setTimeout(() => {
        render();
        resolve();
      }, 10);
    };
    img.onerror = (err) => {
      reject(new Error(`Failed to load image: ${url}`));
    };
    img.src = url;
  });
}

async function loadFrame(frameIndex) {
  if (frameIndex < 0 || frameIndex >= state.frames.length) {
    return;
  }
  
  // Prevent concurrent frame loading to avoid race conditions
  if (state.isNavigating) {
    console.log(`[loadFrame] Skipping navigation to frame ${frameIndex} - already navigating`);
    return;
  }
  
  state.isNavigating = true;
  
  try {
    // Save current frame annotations before switching
    if (state.currentFrameIndex !== frameIndex && (state.studyId || state.tempStudyId)) {
      await saveCurrentFrameAnnotations();
    }
    
    // --- FIX: CLEAR ANNOTATIONS TO PREVENT LEAKAGE ---
    // We must clear the annotation state before updating the frame index.
    // Otherwise, if the network is slow or the user navigates rapidly, 
    // the app might save the OLD annotations to the NEW frame index.
    resetAnnotations();
    // -------------------------------------------------

    state.currentFrameIndex = frameIndex;
  const frame = state.frames[frameIndex];
  
  // Load frame image
  const imageUrl = `/video/${state.videoId}/frame/${frameIndex}`;
  await loadImage(imageUrl);
  
  // Load keypoints/tracks for this frame (only if keypoints file was uploaded)
  if (state.hasKeypointsFile) {
    await loadFrameKeypoints(frameIndex);
  } else {
    // Clear keypoints tracks for this frame if no keypoints file
    state.keypointsTracks[frameIndex] = [];
    state.keypointsTracks[String(frameIndex)] = [];
  }
  
  // Load annotations for this frame
  await loadFrameAnnotations(frameIndex);
  
  updateFrameInfo();
  
  // Force a render after a brief delay to ensure all data is loaded
  setTimeout(() => {
    console.log(`[loadFrame] Triggering render for frame ${frameIndex}`);
    render();
  }, 50);
  } finally {
    state.isNavigating = false;
  }
}

async function loadFrameKeypoints(frameIndex) {
  // Always try to fetch from server to ensure we have the latest data
  // The initial upload data might not be complete or properly structured
  try {
    const url = `/video/${state.videoId}/keypoints/${frameIndex}`;
    console.log(`[loadFrameKeypoints] Fetching from: ${url}`);
    const res = await fetch(url);
    
    if (res.ok) {
      const data = await res.json();
      console.log(`[loadFrameKeypoints] Response data:`, data);
      const tracks = data.tracks || [];
      
      // Store with both number and string keys to handle JSON key issues
      state.keypointsTracks[frameIndex] = tracks;
      state.keypointsTracks[String(frameIndex)] = tracks;
      
      if (tracks.length > 0) {
        console.log(`[loadFrameKeypoints] Loaded ${tracks.length} tracks for frame ${frameIndex} from server`);
        console.log(`[loadFrameKeypoints] First track sample:`, JSON.stringify(tracks[0], null, 2));
        console.log(`[loadFrameKeypoints] Track structure:`, {
          track_id: tracks[0].track_id,
          bbox: tracks[0].bbox,
          keypoints_length: (tracks[0].keypoints || tracks[0].kpts || []).length
        });
      } else {
        console.warn(`[loadFrameKeypoints] No tracks found for frame ${frameIndex}. Response:`, data);
      }
    } else {
      const errorText = await res.text();
      console.warn(`[loadFrameKeypoints] Failed to fetch keypoints for frame ${frameIndex}: ${res.status} - ${errorText}`);
    }
  } catch (error) {
    console.error("[loadFrameKeypoints] Error loading frame keypoints:", error);
  }
}

async function loadFrameAnnotations(frameIndex) {
  const studyIdToUse = state.studyId || state.tempStudyId;
  if (!studyIdToUse) {
    resetAnnotations();
    return;
  }
  
  try {
    const res = await fetch(`/study/${encodeURIComponent(studyIdToUse)}/frame/${frameIndex}/annotations`);
    if (res.ok) {
      const data = await res.json();
      // Load annotations from temp file (these persist across frame navigation)
      state.keypoints = data.keypoints || [];
      state.lines = data.lines || [];
      state.rois = data.rois || [];
      state.measurements = data.measurements || { distances: [], angles: [] };
      
      console.log(`[loadFrameAnnotations] Loaded frame ${frameIndex}: keypoints=${state.keypoints.length}, lines=${state.lines.length}, rois=${state.rois.length}, distances=${state.measurements.distances.length}, angles=${state.measurements.angles.length}`);
      
      // Load bounding box annotations for this frame
      // bounding_boxes contains labels (name, action) per track_id for this frame
      // IMPORTANT: Clear bounding boxes first, then only add the ones from this frame
      // This prevents annotations from previous frames from persisting
      state.boundingBoxes = {};
      if (data.bounding_boxes && Object.keys(data.bounding_boxes).length > 0) {
        // Update state.boundingBoxes with labels from this frame
        // This ensures labels are available for rendering
        for (const [trackId, bboxData] of Object.entries(data.bounding_boxes)) {
          state.boundingBoxes[trackId] = {
            name: bboxData.name || null,
            action: bboxData.action || null,
            annotations: bboxData.annotations || null,
          };
        }
      }
      
      state.nextKeypointIndex = (state.keypoints?.length || 0) + 1;
      state.nextLineIndex = (state.lines?.length || 0) + 1;
      state.nextRoiIndex = (state.rois?.length || 0) + 1;
      state.nextDistanceIndex = (state.measurements.distances?.length || 0) + 1;
      state.nextAngleIndex = (state.measurements.angles?.length || 0) + 1;
      
      updateLists();
      updateBboxInfo();
    } else {
      resetAnnotations();
    }
  } catch (error) {
    console.error("Error loading frame annotations:", error);
    resetAnnotations();
  }
}

function resetAnnotations() {
  state.keypoints = [];
  state.lines = [];
  state.rois = [];
  state.measurements = { distances: [], angles: [] };
  state.boundingBoxes = {};
  state.selectedTrackId = null;
  state.nextKeypointIndex = 1;
  state.nextLineIndex = 1;
  state.nextRoiIndex = 1;
  state.nextDistanceIndex = 1;
  state.nextAngleIndex = 1;
  updateLists();
  updateBboxInfo();
}

function updateFrameInfo() {
  if (frameNumberDisplay) {
    frameNumberDisplay.textContent = state.currentFrameIndex;
  }
  if (frameTotalDisplay) {
    frameTotalDisplay.textContent = state.frames.length - 1;
  }
  if (prevFrameBtn) prevFrameBtn.disabled = state.currentFrameIndex === 0;
  if (nextFrameBtn) nextFrameBtn.disabled = state.currentFrameIndex >= state.frames.length - 1;
}

function setupFrameNumberEditor() {
  if (!frameNumberDisplay) return;
  
  frameNumberDisplay.addEventListener("click", () => {
    const maxFrame = state.frames.length - 1;
    const currentFrame = state.currentFrameIndex;
    
    // Create input element
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = maxFrame.toString();
    input.value = currentFrame.toString();
    input.style.cssText = `
      width: 60px;
      font-size: 0.7rem;
      font-weight: bold;
      text-align: center;
      border: 1px solid #2196f3;
      border-radius: 2px;
      padding: 2px 4px;
      margin: 0 2px;
    `;
    
    // Replace the display with input
    const parent = frameNumberDisplay.parentElement;
    const displayText = frameNumberDisplay.textContent;
    frameNumberDisplay.style.display = "none";
    parent.insertBefore(input, frameNumberDisplay.nextSibling);
    
    // Focus and select
    input.focus();
    input.select();
    
    // Handle Enter key
    const handleEnter = (e) => {
      if (e.key === "Enter") {
        const newFrame = parseInt(input.value, 10);
        if (!isNaN(newFrame) && newFrame >= 0 && newFrame <= maxFrame) {
          loadFrame(newFrame);
        }
        // Remove input and restore display
        input.remove();
        frameNumberDisplay.style.display = "";
      }
    };
    
    // Handle Escape key
    const handleEscape = (e) => {
      if (e.key === "Escape") {
        input.remove();
        frameNumberDisplay.style.display = "";
      }
    };
    
    // Handle blur (click outside)
    const handleBlur = () => {
      const newFrame = parseInt(input.value, 10);
      if (!isNaN(newFrame) && newFrame >= 0 && newFrame <= maxFrame) {
        loadFrame(newFrame);
      }
      if (input.parentNode) { // Check if it still exists
        input.remove();
        frameNumberDisplay.style.display = "";
      }
    };
    
    input.addEventListener("keydown", handleEnter);
    input.addEventListener("keydown", handleEscape);
    input.addEventListener("blur", handleBlur);
  });
}

function updateBboxInfo() {
  if (!bboxInfo) return;
  
  if (state.selectedTrackId && state.boundingBoxes[state.selectedTrackId]) {
    const bbox = state.boundingBoxes[state.selectedTrackId];
    bboxInfo.innerHTML = `
      <p style="margin: 0; font-weight: bold;">Track ID: ${state.selectedTrackId}</p>
      <p style="margin: 0.25rem 0 0 0; color: #666;">Name: ${bbox.name || "Not set"}</p>
      <p style="margin: 0.25rem 0 0 0; color: #666;">Action: ${bbox.action || "Not set"}</p>
    `;
    if (playerNameInput) playerNameInput.value = bbox.name || "";
    if (actionInput) actionInput.value = bbox.action || "";
  } else {
    bboxInfo.innerHTML = `<p style="margin: 0; color: #666;">No bounding box selected</p>`;
    if (playerNameInput) playerNameInput.value = "";
    if (actionInput) actionInput.value = "";
  }
}

function findBoundingBoxAt(worldX, worldY) {
  const frameIdx = state.currentFrameIndex;
  const frameTracks = state.keypointsTracks[frameIdx] || 
                      state.keypointsTracks[String(frameIdx)] || 
                      state.keypointsTracks[frameIdx.toString()] || [];
  
  // Check tracks in reverse order (top-most first) for better selection
  for (let i = frameTracks.length - 1; i >= 0; i--) {
    const track = frameTracks[i];
    const bbox = track.bbox || track.bounding_box;
    if (!bbox || bbox.length !== 4) continue;
    
    // bbox format from file: [x1, y1, x2, y2] (always)
    const [x1, y1, x2, y2] = bbox;
    
    // Ensure x1 < x2 and y1 < y2
    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2);
    const maxY = Math.max(y1, y2);
    
    if (worldX >= minX && worldX <= maxX && worldY >= minY && worldY <= maxY) {
      return track;
    }
  }
  return null;
}

function drawBoundingBoxes() {
  if (!state.keypointsTracks || typeof state.currentFrameIndex !== 'number') {
    console.log(`[drawBoundingBoxes] Early return: keypointsTracks=${!!state.keypointsTracks}, frameIndex=${state.currentFrameIndex}`);
    return; // No data or invalid frame index
  }
  
  // Try both string and number keys for frame index (JSON keys might be strings)
  const frameIdx = state.currentFrameIndex;
  const frameTracks = state.keypointsTracks[frameIdx] || 
                      state.keypointsTracks[String(frameIdx)] || 
                      state.keypointsTracks[frameIdx.toString()] || [];
  
  if (frameTracks.length === 0) {
    return; // Nothing to draw
  }
  
  if (!ctx || !canvas) {
    console.error("[drawBoundingBoxes] Canvas or context is null!", { ctx: !!ctx, canvas: !!canvas });
    return;
  }
  
  // Verify canvas is ready
  if (canvas.width === 0 || canvas.height === 0) {
    console.warn("[drawBoundingBoxes] Canvas has zero dimensions, skipping draw");
    return;
  }
  
  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);
  
  let drawnCount = 0;
  let skippedCount = 0;
  for (const track of frameTracks) {
    const trackId = track.track_id || track.id;
    const bbox = track.bbox || track.bounding_box;
    
    if (!bbox || bbox.length !== 4) {
      skippedCount++;
      continue;
    }
    
    // bbox format: [x1, y1, x2, y2]
    const [x1, y1, x2, y2] = bbox;
    
    // Check if this track is selected
    const isSelected = state.selectedTrackId === trackId;
    const bboxData = state.boundingBoxes[trackId] || {};
    
    // Get color for this track (consistent per track_id)
    const trackColor = trackId !== null && trackId !== undefined ? colorForTrackId(trackId) : "#ff0000";
    
    // Draw bounding box
    if (isSelected) {
      // Selected: black, dashed, thick outline
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 4 / state.zoom;
      ctx.setLineDash([8 / state.zoom, 4 / state.zoom]);
    } else {
      // Not selected: track color, solid, normal thickness
      ctx.strokeStyle = trackColor;
      ctx.lineWidth = 2 / state.zoom;
      ctx.setLineDash([]);
    }
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]); // Reset dash pattern
    
    // Draw track_id and score at top-left corner of bbox (always shown)
    const score = track.score || track.conf || 0.0;
    const trackLabelParts = [];
    if (trackId !== null && trackId !== undefined) {
      trackLabelParts.push(`T${trackId}`);
    }
    trackLabelParts.push(score.toFixed(2));
    const trackLabel = trackLabelParts.join(" ");
    
    const trackFontSize = 10 / state.zoom;
    ctx.font = `${trackFontSize}px system-ui`;
    ctx.textBaseline = "bottom";
    
    const trackMetrics = ctx.measureText(trackLabel);
    const trackTextWidth = trackMetrics.width;
    const trackTextHeight = trackFontSize;
    const trackPadding = 2 / state.zoom;
    const trackLabelX = x1;
    const trackLabelY = y1 - trackPadding;
    
    ctx.fillStyle = trackColor;
    ctx.fillRect(
      trackLabelX,
      trackLabelY - trackTextHeight - trackPadding,
      trackTextWidth + trackPadding * 2,
      trackTextHeight + trackPadding * 2
    );
    
    ctx.fillStyle = "#000000";
    ctx.fillText(trackLabel, trackLabelX + trackPadding, trackLabelY);
    
    // Draw name and action label to the left of the bounding box (if present)
    if (bboxData.name || bboxData.action) {
      const fontSize = 14 / state.zoom; // Bigger font
      ctx.font = `bold ${fontSize}px system-ui`;
      ctx.textBaseline = "top";
      
      // Build label text
      const labelParts = [];
      if (bboxData.name) {
        labelParts.push(`Name: ${bboxData.name}`);
      }
      if (bboxData.action) {
        labelParts.push(`Action: ${bboxData.action}`);
      }
      const label = labelParts.join(", ");
      
      // Measure text
      const metrics = ctx.measureText(label);
      const textWidth = metrics.width;
      const textHeight = fontSize;
      const padding = 4 / state.zoom;
      
      // Position to the left of bounding box
      const labelX = x1 - textWidth - padding * 2;
      const labelY = y1;
      
      // Draw label background
      const labelBgColor = isSelected ? "#000000" : trackColor;
      ctx.fillStyle = labelBgColor;
      ctx.fillRect(
        labelX,
        labelY,
        textWidth + padding * 2,
        textHeight + padding * 2
      );
      
      // Draw label text (white if selected/black background, black otherwise)
      ctx.fillStyle = isSelected ? "#ffffff" : "#000000";
      ctx.fillText(label, labelX + padding, labelY + padding);
    }
    
    // Draw keypoints and skeleton if available (format: [x1, y1, conf1, x2, y2, conf2, ...])
    const keypoints = track.keypoints || track.kpts || [];
    if (keypoints.length > 0) {
      const kpThreshold = 0.01; // Minimum confidence to draw keypoint
      const kpRadius = 3 / state.zoom;
      const kpThickness = 2 / state.zoom;
      const kpColor = trackColor; // Use track color for keypoints, not selection color
      
      // Helper to get keypoint at index
      const getKeypoint = (idx) => {
        const base = idx * 3;
        if (base + 2 < keypoints.length) {
          return {
            x: keypoints[base],
            y: keypoints[base + 1],
            conf: keypoints[base + 2] || 0.0
          };
        }
        return null;
      };
      
      // Draw skeleton connections first (so keypoints appear on top)
      ctx.strokeStyle = kpColor;
      ctx.lineWidth = kpThickness;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      
      for (const [i, j] of COCO17_EDGES) {
        const kp1 = getKeypoint(i);
        const kp2 = getKeypoint(j);
        
        if (kp1 && kp2 && kp1.conf >= kpThreshold && kp2.conf >= kpThreshold) {
          ctx.beginPath();
          ctx.moveTo(kp1.x, kp1.y);
          ctx.lineTo(kp2.x, kp2.y);
          ctx.stroke();
        }
      }
      
      // Draw keypoints
      ctx.fillStyle = kpColor;
      for (let i = 0; i < keypoints.length; i += 3) {
        if (i + 2 < keypoints.length) {
          const x = keypoints[i];
          const y = keypoints[i + 1];
          const conf = keypoints[i + 2] || 0.0;
          
          // Only draw if confidence is above threshold
          if (conf >= kpThreshold) {
            ctx.beginPath();
            ctx.arc(x, y, kpRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw outline for better visibility
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 0.5 / state.zoom;
            ctx.stroke();
          }
        }
      }
    }
    
    drawnCount++;
  }
  
  ctx.restore();
}

// Helper functions for rendering
function drawRulerLine(ctx, startX, startY, endX, endY, color, zoom) {
  const dx = endX - startX;
  const dy = endY - startY;
  const length = Math.hypot(dx, dy);
  
  // Draw dotted main line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 / zoom;
  ctx.setLineDash([5 / zoom, 5 / zoom]);
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Draw perpendicular hash marks
  const hashSpacing = 20 / zoom;
  const hashLength = 6 / zoom;
  const numHashes = Math.floor(length / hashSpacing);
  
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 / zoom;
  
  for (let i = 1; i < numHashes; i++) {
    const t = (i * hashSpacing) / length;
    const hx = startX + dx * t;
    const hy = startY + dy * t;
    const perpX = -dy / length;
    const perpY = dx / length;
    
    ctx.beginPath();
    ctx.moveTo(hx - perpX * hashLength, hy - perpY * hashLength);
    ctx.lineTo(hx + perpX * hashLength, hy + perpY * hashLength);
    ctx.stroke();
  }
}

function distancePointToSegment(px, py, a, b) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = px - a.x;
  const wy = py - a.y;
  const c1 = vx * wx + vy * wy;
  if (c1 <= 0) return Math.hypot(px - a.x, py - a.y);
  const c2 = vx * vx + vy * vy;
  if (c2 <= c1) return Math.hypot(px - b.x, py - b.y);
  const t = c1 / c2;
  const projx = a.x + t * vx;
  const projy = a.y + t * vy;
  return Math.hypot(px - projx, py - projy);
}

function drawTriangle(ctx, x, y, size, color, zoom) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 / zoom;
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x - size, y + size);
  ctx.lineTo(x + size, y + size);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawRoiDot(ctx, x, y, radius, color, zoom) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2 / zoom;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.4, 0, 2 * Math.PI);
  ctx.fillStyle = "#fff";
  ctx.fill();
}

function angleFromThreePoints(p1, p2, p3) {
  const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (!mag1 || !mag2) return 0;
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function findNearestKeypoint(worldX, worldY, threshold = 10) {
  let nearest = null;
  let best = threshold;
  for (const kp of state.keypoints) {
    const d = Math.hypot(kp.x - worldX, kp.y - worldY);
    const adjustedThreshold = threshold / state.zoom;
    if (d <= adjustedThreshold && d <= best) {
      best = d;
      nearest = kp;
    }
  }
  return nearest;
}

function findNearestLineEndpoint(worldX, worldY, threshold = 10) {
  let nearest = null;
  let endpoint = null;
  let best = threshold;
  const adjustedThreshold = threshold / state.zoom;
  for (const line of state.lines) {
    const dStart = Math.hypot(line.start.x - worldX, line.start.y - worldY);
    const dEnd = Math.hypot(line.end.x - worldX, line.end.y - worldY);
    if (dStart <= adjustedThreshold && dStart <= best) {
      best = dStart;
      nearest = line;
      endpoint = "start";
    }
    if (dEnd <= adjustedThreshold && dEnd <= best) {
      best = dEnd;
      nearest = line;
      endpoint = "end";
    }
  }
  return nearest ? { line: nearest, endpoint } : null;
}

function findNearestDistanceEndpoint(worldX, worldY, threshold = 10) {
  let nearest = null;
  let endpoint = null;
  let best = threshold;
  const adjustedThreshold = threshold / state.zoom;
  for (const dist of state.measurements.distances) {
    if (!dist.start || !dist.end) continue;
    const dStart = Math.hypot(dist.start.x - worldX, dist.start.y - worldY);
    const dEnd = Math.hypot(dist.end.x - worldX, dist.end.y - worldY);
    if (dStart <= adjustedThreshold && dStart <= best) {
      best = dStart;
      nearest = dist;
      endpoint = "start";
    }
    if (dEnd <= adjustedThreshold && dEnd <= best) {
      best = dEnd;
      nearest = dist;
      endpoint = "end";
    }
  }
  return nearest ? { distance: nearest, endpoint } : null;
}

function findNearestAnglePoint(worldX, worldY, threshold = 10) {
  let nearest = null;
  let pointType = null;
  let best = threshold;
  const adjustedThreshold = threshold / state.zoom;
  for (const ang of state.measurements.angles) {
    if (!ang.point1 || !ang.point2 || !ang.point3) continue;
    const d1 = Math.hypot(ang.point1.x - worldX, ang.point1.y - worldY);
    const d2 = Math.hypot(ang.point2.x - worldX, ang.point2.y - worldY);
    const d3 = Math.hypot(ang.point3.x - worldX, ang.point3.y - worldY);
    if (d1 <= adjustedThreshold && d1 <= best) {
      best = d1;
      nearest = ang;
      pointType = "point1";
    }
    if (d2 <= adjustedThreshold && d2 <= best) {
      best = d2;
      nearest = ang;
      pointType = "point2";
    }
    if (d3 <= adjustedThreshold && d3 <= best) {
      best = d3;
      nearest = ang;
      pointType = "point3";
    }
  }
  return nearest ? { angle: nearest, pointType } : null;
}

function findNearestRoiPoint(worldX, worldY, threshold = 10) {
  let nearest = null;
  let pointIndex = null;
  let best = threshold;
  const adjustedThreshold = threshold / state.zoom;
  for (const roi of state.rois) {
    if (!roi.points || roi.points.length === 0) continue;
    for (let i = 0; i < roi.points.length; i++) {
      const point = roi.points[i];
      const d = Math.hypot(point.x - worldX, point.y - worldY);
      if (d <= adjustedThreshold && d <= best) {
        best = d;
        nearest = roi;
        pointIndex = i;
      }
    }
  }
  return nearest ? { roi: nearest, pointIndex } : null;
}

async function addKeypoint(worldX, worldY) {
  const label = `#${state.nextKeypointIndex}`;
  const color = COLOR_OPTIONS[colorIndex % COLOR_OPTIONS.length];
  colorIndex += 1;
  const id = `kp-${state.nextKeypointIndex}`;
  const keypoint = { id, label, x: worldX, y: worldY, color };
  
  // Associate with selected bounding box if one is selected
  if (state.selectedTrackId) {
    if (!state.boundingBoxes[state.selectedTrackId]) {
      state.boundingBoxes[state.selectedTrackId] = {};
    }
    if (!state.boundingBoxes[state.selectedTrackId].annotations) {
      state.boundingBoxes[state.selectedTrackId].annotations = { keypoints: [], lines: [], rois: [], measurements: { distances: [], angles: [] } };
    }
    state.boundingBoxes[state.selectedTrackId].annotations.keypoints.push(keypoint);
  }
  
  state.keypoints.push(keypoint);
  state.nextKeypointIndex += 1;
  updateLists();
  render();
  
  // Auto-save annotations immediately to temp file
  // This ensures annotations persist while navigating frames
  await saveCurrentFrameAnnotations();
  console.log(`[addKeypoint] Saved keypoint to temp file for frame ${state.currentFrameIndex}`);
}

function startAddKeypointMode() {
  setMode("add-keypoint", `Click on canvas to place keypoint ${state.nextKeypointIndex}.`);
}

function startDrawLineMode() {
  state.tempPoints = [];
  setMode("draw-line", `Click two points (or keypoints) to draw line ${state.nextLineIndex}.`);
}

function startRoiMode() {
  state.tempPoints = [];
  setMode("roi", `Click at least 3 points to create ROI ${state.nextRoiIndex}. Press Enter to complete.`);
}

function startDistanceMode() {
  state.tempPoints = [];
  setMode("distance", `Click two points to measure distance ${state.nextDistanceIndex}.`);
}

function startAngleMode() {
  state.tempPoints = [];
  setMode("angle", `Click three points to measure angle ${state.nextAngleIndex}.`);
}

async function completeRoi() {
  if (state.tempPoints.length < 3) {
    alert("ROI must have at least 3 points.");
    return;
  }
  const label = `#${state.nextRoiIndex}`;
  const color = COLOR_OPTIONS[roiColorIndex % COLOR_OPTIONS.length];
  roiColorIndex += 1;
  const id = `roi-${state.nextRoiIndex}`;
  const points = state.tempPoints.map(p => ({ x: p.x, y: p.y }));
  const roi = { id, label, points, color };
  
  // Associate with selected bounding box if one is selected
  if (state.selectedTrackId) {
    if (!state.boundingBoxes[state.selectedTrackId]) {
      state.boundingBoxes[state.selectedTrackId] = {};
    }
    if (!state.boundingBoxes[state.selectedTrackId].annotations) {
      state.boundingBoxes[state.selectedTrackId].annotations = { keypoints: [], lines: [], rois: [], measurements: { distances: [], angles: [] } };
    }
    state.boundingBoxes[state.selectedTrackId].annotations.rois.push(roi);
  }
  
  state.rois.push(roi);
  state.nextRoiIndex += 1;
  state.tempPoints = [];
  updateLists();
  render();
  setMode("idle", `ROI ${label} added.`);
  
  // Auto-save annotations
  await saveCurrentFrameAnnotations();
}

function deleteKeypoint(id) {
  state.keypoints = state.keypoints.filter((k) => k.id !== id);
  updateLists();
  render();
}

function deleteLine(id) {
  state.lines = state.lines.filter((l) => l.id !== id);
  updateLists();
  render();
}

function deleteDistance(id) {
  state.measurements.distances = state.measurements.distances.filter((d) => d.id !== id);
  updateLists();
  render();
}

function deleteAngle(id) {
  state.measurements.angles = state.measurements.angles.filter((a) => a.id !== id);
  updateLists();
  render();
}

function deleteRoi(id) {
  state.rois = state.rois.filter((r) => r.id !== id);
  updateLists();
  render();
}

function createDeleteButton(onClick) {
  const deleteBtn = document.createElement("button");
  deleteBtn.innerHTML = "×";
  deleteBtn.style.flexShrink = "0";
  deleteBtn.style.width = "24px";
  deleteBtn.style.height = "24px";
  deleteBtn.style.minWidth = "24px";
  deleteBtn.style.maxWidth = "24px";
  deleteBtn.style.minHeight = "24px";
  deleteBtn.style.maxHeight = "24px";
  deleteBtn.style.padding = "0";
  deleteBtn.style.margin = "0";
  deleteBtn.style.backgroundColor = "#dc3545";
  deleteBtn.style.color = "#fff";
  deleteBtn.style.border = "1px solid #c82333";
  deleteBtn.style.borderRadius = "2px";
  deleteBtn.style.cursor = "pointer";
  deleteBtn.style.fontSize = "16px";
  deleteBtn.style.fontWeight = "bold";
  deleteBtn.style.lineHeight = "1";
  deleteBtn.style.display = "flex";
  deleteBtn.style.alignItems = "center";
  deleteBtn.style.justifyContent = "center";
  deleteBtn.addEventListener("click", onClick);
  return deleteBtn;
}

function createRenameButton(onClick) {
  const renameBtn = document.createElement("button");
  renameBtn.innerHTML = "✎";
  renameBtn.style.flexShrink = "0";
  renameBtn.style.width = "24px";
  renameBtn.style.height = "24px";
  renameBtn.style.minWidth = "24px";
  renameBtn.style.maxWidth = "24px";
  renameBtn.style.minHeight = "24px";
  renameBtn.style.maxHeight = "24px";
  renameBtn.style.padding = "0";
  renameBtn.style.margin = "0";
  renameBtn.style.marginLeft = "auto";
  renameBtn.style.backgroundColor = "#007bff";
  renameBtn.style.color = "#fff";
  renameBtn.style.border = "1px solid #0056b3";
  renameBtn.style.borderRadius = "2px";
  renameBtn.style.cursor = "pointer";
  renameBtn.style.fontSize = "14px";
  renameBtn.style.fontWeight = "normal";
  renameBtn.style.lineHeight = "1";
  renameBtn.style.display = "flex";
  renameBtn.style.alignItems = "center";
  renameBtn.style.justifyContent = "center";
  renameBtn.addEventListener("click", onClick);
  return renameBtn;
}

function renameItem(item, itemType) {
  const currentLabel = item.label || item.id;
  const newLabel = prompt(`Enter new name for ${itemType}:`, currentLabel);
  if (newLabel !== null && newLabel.trim() !== "") {
    item.label = newLabel.trim();
    updateLists();
    render();
  }
}

function truncateLabel(label, maxLength = 5) {
  if (!label || label.length <= maxLength) {
    return label || "";
  }
  return label.substring(0, maxLength) + "...";
}

function showColorPicker(item) {
  let picker = document.getElementById(`color-picker-${item.id}`);
  if (picker) {
    picker.remove();
    return;
  }

  picker = document.createElement("div");
  picker.id = `color-picker-${item.id}`;
  picker.className = "color-picker-popup";
  picker.style.position = "absolute";
  picker.style.background = "#fff";
  picker.style.border = "1px solid #ccc";
  picker.style.borderRadius = "4px";
  picker.style.padding = "0.5rem";
  picker.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";
  picker.style.zIndex = "1000";
  picker.style.display = "grid";
  picker.style.gridTemplateColumns = "repeat(5, 1fr)";
  picker.style.gap = "0.25rem";

  COLOR_OPTIONS.forEach((color) => {
    const swatch = document.createElement("button");
    swatch.className = "color-swatch";
    swatch.style.width = "24px";
    swatch.style.height = "24px";
    swatch.style.backgroundColor = color;
    swatch.style.border = color === item.color ? "2px solid #000" : "1px solid #ccc";
    swatch.style.borderRadius = "4px";
    swatch.style.cursor = "pointer";
    swatch.style.padding = "0";
    swatch.addEventListener("click", () => {
      item.color = color;
      render();
      updateLists();
    });
    picker.appendChild(swatch);
  });

  const colorBox = document.getElementById(`color-box-${item.id}`);
  if (colorBox) {
    const rect = colorBox.getBoundingClientRect();
    picker.style.left = `${rect.left}px`;
    picker.style.top = `${rect.bottom + 5}px`;
  }

  document.body.appendChild(picker);

  const closePicker = (e) => {
    if (!picker.contains(e.target) && e.target !== colorBox) {
      picker.remove();
      document.removeEventListener("click", closePicker);
    }
  };
  setTimeout(() => document.addEventListener("click", closePicker), 0);
}

function updateLists() {
  keypointList.innerHTML = "";
  for (const kp of state.keypoints) {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.gap = "0.5rem";
    li.style.padding = "0.5rem 0";
    
    const labelText = document.createElement("span");
    labelText.textContent = `${kp.label}`;
    labelText.style.flexShrink = "0";
    li.appendChild(labelText);
    
    const coordsText = document.createElement("span");
    coordsText.textContent = `(${Math.round(kp.x)}, ${Math.round(kp.y)})`;
    coordsText.style.flexShrink = "0";
    coordsText.style.color = "#666";
    coordsText.style.fontSize = "0.9rem";
    li.appendChild(coordsText);
    
    const colorBox = document.createElement("button");
    colorBox.id = `color-box-${kp.id}`;
    colorBox.className = "keypoint-color-box";
    colorBox.style.width = "16px";
    colorBox.style.height = "16px";
    colorBox.style.minWidth = "16px";
    colorBox.style.maxWidth = "16px";
    colorBox.style.backgroundColor = kp.color;
    colorBox.style.border = "1px solid #333";
    colorBox.style.borderRadius = "2px";
    colorBox.style.cursor = "pointer";
    colorBox.style.padding = "0";
    colorBox.style.flexShrink = "0";
    colorBox.style.flexGrow = "0";
    colorBox.addEventListener("click", (e) => {
      e.stopPropagation();
      showColorPicker(kp);
    });
    li.appendChild(colorBox);
    
    const renameBtn = createRenameButton(() => renameItem(kp, "keypoint"));
    li.appendChild(renameBtn);
    
    const deleteBtn = createDeleteButton(() => deleteKeypoint(kp.id));
    li.appendChild(deleteBtn);

    keypointList.appendChild(li);
  }

  lineList.innerHTML = "";
  for (const line of state.lines) {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.gap = "0.5rem";
    li.style.padding = "0.5rem 0";
    
    const labelText = document.createElement("span");
    labelText.textContent = truncateLabel(line.label || line.id);
    labelText.style.flexShrink = "0";
    labelText.style.minWidth = "0";
    labelText.style.overflow = "hidden";
    labelText.style.textOverflow = "ellipsis";
    li.appendChild(labelText);
    
    const infoText = document.createElement("span");
    infoText.textContent = `(${Math.round(line.start.x)}, ${Math.round(line.start.y)}): (${Math.round(line.end.x)}, ${Math.round(line.end.y)})`;
    infoText.style.flexShrink = "0";
    infoText.style.color = "#666";
    infoText.style.fontSize = "0.9rem";
    li.appendChild(infoText);
    
    const colorBox = document.createElement("button");
    colorBox.id = `color-box-${line.id}`;
    colorBox.className = "keypoint-color-box";
    colorBox.style.width = "16px";
    colorBox.style.height = "16px";
    colorBox.style.minWidth = "16px";
    colorBox.style.maxWidth = "16px";
    colorBox.style.backgroundColor = line.color || "#4fc3f7";
    colorBox.style.border = "1px solid #333";
    colorBox.style.borderRadius = "2px";
    colorBox.style.cursor = "pointer";
    colorBox.style.padding = "0";
    colorBox.style.flexShrink = "0";
    colorBox.style.flexGrow = "0";
    colorBox.addEventListener("click", (e) => {
      e.stopPropagation();
      showColorPicker(line);
    });
    li.appendChild(colorBox);
    
    const renameBtn = createRenameButton(() => renameItem(line, "line"));
    li.appendChild(renameBtn);
    
    const deleteBtn = createDeleteButton(() => deleteLine(line.id));
    li.appendChild(deleteBtn);

    lineList.appendChild(li);
  }

  roiList.innerHTML = "";
  for (const roi of state.rois) {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.gap = "0.5rem";
    li.style.padding = "0.5rem 0";
    
    const labelText = document.createElement("span");
    labelText.textContent = truncateLabel(roi.label || roi.id);
    labelText.style.flexShrink = "0";
    labelText.style.minWidth = "0";
    labelText.style.overflow = "hidden";
    labelText.style.textOverflow = "ellipsis";
    li.appendChild(labelText);
    
    const infoText = document.createElement("span");
    infoText.textContent = `${roi.points?.length || 0} points`;
    infoText.style.flexShrink = "0";
    infoText.style.color = "#666";
    infoText.style.fontSize = "0.9rem";
    li.appendChild(infoText);
    
    const colorBox = document.createElement("button");
    colorBox.id = `color-box-${roi.id}`;
    colorBox.className = "keypoint-color-box";
    colorBox.style.width = "16px";
    colorBox.style.height = "16px";
    colorBox.style.minWidth = "16px";
    colorBox.style.maxWidth = "16px";
    colorBox.style.backgroundColor = roi.color || "#4fc3f7";
    colorBox.style.border = "1px solid #333";
    colorBox.style.borderRadius = "2px";
    colorBox.style.cursor = "pointer";
    colorBox.style.padding = "0";
    colorBox.style.flexShrink = "0";
    colorBox.style.flexGrow = "0";
    colorBox.addEventListener("click", (e) => {
      e.stopPropagation();
      showColorPicker(roi);
    });
    li.appendChild(colorBox);
    
    const renameBtn = createRenameButton(() => renameItem(roi, "ROI"));
    li.appendChild(renameBtn);
    
    const deleteBtn = createDeleteButton(() => deleteRoi(roi.id));
    li.appendChild(deleteBtn);

    roiList.appendChild(li);
  }

  distanceList.innerHTML = "";
  for (const dist of state.measurements.distances) {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.gap = "0.5rem";
    li.style.padding = "0.5rem 0";
    
    const labelText = document.createElement("span");
    labelText.textContent = truncateLabel(dist.label || dist.id);
    labelText.style.flexShrink = "0";
    labelText.style.minWidth = "0";
    labelText.style.overflow = "hidden";
    labelText.style.textOverflow = "ellipsis";
    li.appendChild(labelText);
    
    const infoText = document.createElement("span");
    if (dist.start && dist.end) {
      infoText.textContent = `${Math.round(dist.value)} px`;
    } else {
      infoText.textContent = "Invalid";
    }
    infoText.style.flexShrink = "0";
    infoText.style.color = "#666";
    infoText.style.fontSize = "0.9rem";
    li.appendChild(infoText);
    
    const colorBox = document.createElement("button");
    colorBox.id = `color-box-${dist.id}`;
    colorBox.className = "keypoint-color-box";
    colorBox.style.width = "16px";
    colorBox.style.height = "16px";
    colorBox.style.minWidth = "16px";
    colorBox.style.maxWidth = "16px";
    colorBox.style.backgroundColor = dist.color || "#4fc3f7";
    colorBox.style.border = "1px solid #333";
    colorBox.style.borderRadius = "2px";
    colorBox.style.cursor = "pointer";
    colorBox.style.padding = "0";
    colorBox.style.flexShrink = "0";
    colorBox.style.flexGrow = "0";
    colorBox.addEventListener("click", (e) => {
      e.stopPropagation();
      showColorPicker(dist);
    });
    li.appendChild(colorBox);
    
    const renameBtn = createRenameButton(() => renameItem(dist, "distance"));
    li.appendChild(renameBtn);
    
    const deleteBtn = createDeleteButton(() => deleteDistance(dist.id));
    li.appendChild(deleteBtn);

    distanceList.appendChild(li);
  }

  angleList.innerHTML = "";
  for (const ang of state.measurements.angles) {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.gap = "0.5rem";
    li.style.padding = "0.5rem 0";
    
    const labelText = document.createElement("span");
    labelText.textContent = truncateLabel(ang.label || ang.id);
    labelText.style.flexShrink = "0";
    labelText.style.minWidth = "0";
    labelText.style.overflow = "hidden";
    labelText.style.textOverflow = "ellipsis";
    li.appendChild(labelText);
    
    const infoText = document.createElement("span");
    const displayedAngle = ang.isReflex ? (360 - ang.value) : ang.value;
    infoText.textContent = `${Math.round(displayedAngle)}°`;
    infoText.style.flexShrink = "0";
    infoText.style.color = "#666";
    infoText.style.fontSize = "0.9rem";
    li.appendChild(infoText);
    
    const colorBox = document.createElement("button");
    colorBox.id = `color-box-${ang.id}`;
    colorBox.className = "keypoint-color-box";
    colorBox.style.width = "16px";
    colorBox.style.height = "16px";
    colorBox.style.minWidth = "16px";
    colorBox.style.maxWidth = "16px";
    colorBox.style.backgroundColor = ang.color || "#4fc3f7";
    colorBox.style.border = "1px solid #333";
    colorBox.style.borderRadius = "2px";
    colorBox.style.cursor = "pointer";
    colorBox.style.padding = "0";
    colorBox.style.flexShrink = "0";
    colorBox.style.flexGrow = "0";
    colorBox.addEventListener("click", (e) => {
      e.stopPropagation();
      showColorPicker(ang);
    });
    li.appendChild(colorBox);
    
    const renameBtn = createRenameButton(() => renameItem(ang, "angle"));
    li.appendChild(renameBtn);
    
    const deleteBtn = createDeleteButton(() => deleteAngle(ang.id));
    li.appendChild(deleteBtn);

    angleList.appendChild(li);
  }
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  if (state.image) {
    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);
    ctx.drawImage(state.image, 0, 0);
    ctx.restore();
  }

  // Draw bounding boxes and keypoints FIRST (so annotations appear on top)
  drawBoundingBoxes();
  
  // Draw ROIs
  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);
  
  for (const roi of state.rois) {
    if (!roi.points || roi.points.length === 0) continue;
    const roiColor = roi.color || "#4fc3f7";
    const dotRadius = KEYPOINT_SIZE / state.zoom;
    
    for (const point of roi.points) {
      drawRoiDot(ctx, point.x, point.y, dotRadius, roiColor, state.zoom);
    }
    
    ctx.strokeStyle = roiColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
    
    for (let i = 0; i < roi.points.length; i++) {
      const current = roi.points[i];
      const next = roi.points[(i + 1) % roi.points.length];
      ctx.beginPath();
      ctx.moveTo(current.x, current.y);
      ctx.lineTo(next.x, next.y);
      ctx.stroke();
    }
    
    ctx.setLineDash([]);
  }
  
  ctx.restore();
  
  // Draw lines
  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);
  
  // Preview line
  if (state.mode === "draw-line" && state.tempPoints.length === 1 && state.previewMouseX !== null && state.previewMouseY !== null) {
    const start = state.tempPoints[0];
    const previewColor = COLOR_OPTIONS[lineColorIndex % COLOR_OPTIONS.length];
    ctx.strokeStyle = previewColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(state.previewMouseX, state.previewMouseY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  
  // Preview distance
  if (state.mode === "distance" && state.tempPoints.length === 1 && state.previewMouseX !== null && state.previewMouseY !== null) {
    const start = state.tempPoints[0];
    const previewColor = COLOR_OPTIONS[distanceColorIndex % COLOR_OPTIONS.length];
    drawRulerLine(ctx, start.x, start.y, state.previewMouseX, state.previewMouseY, previewColor, state.zoom);
  }
  
  // Preview ROI
  if (state.mode === "roi" && state.tempPoints.length >= 1 && state.previewMouseX !== null && state.previewMouseY !== null) {
    const previewColor = COLOR_OPTIONS[roiColorIndex % COLOR_OPTIONS.length];
    const dotRadius = KEYPOINT_SIZE / state.zoom;
    
    for (let i = 0; i < state.tempPoints.length; i++) {
      const point = state.tempPoints[i];
      drawRoiDot(ctx, point.x, point.y, dotRadius, previewColor, state.zoom);
      
      if (i < state.tempPoints.length - 1) {
        const nextPoint = state.tempPoints[i + 1];
        ctx.strokeStyle = previewColor;
        ctx.lineWidth = 2 / state.zoom;
        ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(nextPoint.x, nextPoint.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    
    if (state.tempPoints.length > 0) {
      const lastPoint = state.tempPoints[state.tempPoints.length - 1];
      ctx.strokeStyle = previewColor;
      ctx.lineWidth = 2 / state.zoom;
      ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(state.previewMouseX, state.previewMouseY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  
  // Preview angle
  if (state.mode === "angle" && state.tempPoints.length >= 1 && state.previewMouseX !== null && state.previewMouseY !== null) {
    const previewColor = COLOR_OPTIONS[angleColorIndex % COLOR_OPTIONS.length];
    const triangleSize = 8 / state.zoom;
    
    if (state.tempPoints.length >= 1) {
      const p1 = state.tempPoints[0];
      drawTriangle(ctx, p1.x, p1.y, triangleSize, previewColor, state.zoom);
      
      if (state.tempPoints.length === 1) {
        ctx.strokeStyle = previewColor;
        ctx.lineWidth = 2 / state.zoom;
        ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(state.previewMouseX, state.previewMouseY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    
    if (state.tempPoints.length >= 2) {
      const p1 = state.tempPoints[0];
      const p2 = state.tempPoints[1];
      drawTriangle(ctx, p2.x, p2.y, triangleSize, previewColor, state.zoom);
      
      ctx.strokeStyle = previewColor;
      ctx.lineWidth = 2 / state.zoom;
      ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.setLineDash([]);
      
      if (state.tempPoints.length === 2) {
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(state.previewMouseX, state.previewMouseY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        const previewAngle = angleFromThreePoints(p1, p2, { x: state.previewMouseX, y: state.previewMouseY });
        const vertex = p2;
        const arcRadius = 20 / state.zoom;
        
        let angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
        let angle2 = Math.atan2(state.previewMouseY - vertex.y, state.previewMouseX - vertex.x);
        
        angle1 = ((angle1 % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
        angle2 = ((angle2 % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
        
        let diff = angle2 - angle1;
        if (diff < 0) diff += 2 * Math.PI;
        
        if (diff > Math.PI) {
          [angle1, angle2] = [angle2, angle1];
        }
        
        ctx.strokeStyle = previewColor;
        ctx.lineWidth = 2 / state.zoom;
        ctx.beginPath();
        ctx.arc(vertex.x, vertex.y, arcRadius, angle1, angle2, false);
        ctx.stroke();
        
        ctx.fillStyle = previewColor;
        ctx.font = `${12 / state.zoom}px system-ui`;
        ctx.fillText(`${Math.round(previewAngle)}°`, vertex.x + 8 / state.zoom, vertex.y - 8 / state.zoom);
      }
    }
  }
  
  // Draw lines
  for (const line of state.lines) {
    const lineColor = line.color || "#4fc3f7";
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.beginPath();
    ctx.moveTo(line.start.x, line.start.y);
    ctx.lineTo(line.end.x, line.end.y);
    ctx.stroke();
    
    const xSize = 8 / state.zoom;
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2 / state.zoom;
    
    ctx.beginPath();
    ctx.moveTo(line.start.x - xSize, line.start.y - xSize);
    ctx.lineTo(line.start.x + xSize, line.start.y + xSize);
    ctx.moveTo(line.start.x - xSize, line.start.y + xSize);
    ctx.lineTo(line.start.x + xSize, line.start.y - xSize);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(line.end.x - xSize, line.end.y - xSize);
    ctx.lineTo(line.end.x + xSize, line.end.y + xSize);
    ctx.moveTo(line.end.x - xSize, line.end.y + xSize);
    ctx.lineTo(line.end.x + xSize, line.end.y - xSize);
    ctx.stroke();
  }
  
  // Draw X mark at starting point when drawing a line
  if (state.mode === "draw-line" && state.tempPoints.length === 1) {
    const start = state.tempPoints[0];
    const previewColor = COLOR_OPTIONS[lineColorIndex % COLOR_OPTIONS.length];
    const xSize = 8 / state.zoom;
    ctx.strokeStyle = previewColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.beginPath();
    ctx.moveTo(start.x - xSize, start.y - xSize);
    ctx.lineTo(start.x + xSize, start.y + xSize);
    ctx.moveTo(start.x - xSize, start.y + xSize);
    ctx.lineTo(start.x + xSize, start.y - xSize);
    ctx.stroke();
  }

  // Draw distances
  for (const dist of state.measurements.distances) {
    if (!dist.start || !dist.end) continue;
    const distColor = dist.color || "#4fc3f7";
    
    drawRulerLine(ctx, dist.start.x, dist.start.y, dist.end.x, dist.end.y, distColor, state.zoom);
    
    const squareSize = 8 / state.zoom;
    ctx.fillStyle = distColor;
    ctx.strokeStyle = distColor;
    ctx.lineWidth = 2 / state.zoom;
    
    ctx.fillRect(dist.start.x - squareSize / 2, dist.start.y - squareSize / 2, squareSize, squareSize);
    ctx.strokeRect(dist.start.x - squareSize / 2, dist.start.y - squareSize / 2, squareSize, squareSize);
    
    ctx.fillRect(dist.end.x - squareSize / 2, dist.end.y - squareSize / 2, squareSize, squareSize);
    ctx.strokeRect(dist.end.x - squareSize / 2, dist.end.y - squareSize / 2, squareSize, squareSize);
  }
  
  // Draw square at starting point when measuring distance
  if (state.mode === "distance" && state.tempPoints.length === 1) {
    const start = state.tempPoints[0];
    const previewColor = COLOR_OPTIONS[distanceColorIndex % COLOR_OPTIONS.length];
    const squareSize = 8 / state.zoom;
    ctx.fillStyle = previewColor;
    ctx.strokeStyle = previewColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.fillRect(start.x - squareSize / 2, start.y - squareSize / 2, squareSize, squareSize);
    ctx.strokeRect(start.x - squareSize / 2, start.y - squareSize / 2, squareSize, squareSize);
  }

  // Draw keypoints
  for (const kp of state.keypoints) {
    const kpColor = kp.color || "#ff5252";
    const radius = KEYPOINT_SIZE / state.zoom;
    ctx.fillStyle = kpColor;
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.fillStyle = "#fff";
    ctx.font = `${12 / state.zoom}px system-ui`;
    ctx.fillText(kp.label, kp.x + 8 / state.zoom, kp.y - 8 / state.zoom);
  }
  
  // Draw angles
  for (const ang of state.measurements.angles) {
    if (!ang.point1 || !ang.point2 || !ang.point3) continue;
    const angColor = ang.color || "#4fc3f7";
    const triangleSize = 8 / state.zoom;
    
    // Draw triangles
    drawTriangle(ctx, ang.point1.x, ang.point1.y, triangleSize, angColor, state.zoom);
    drawTriangle(ctx, ang.point2.x, ang.point2.y, triangleSize, angColor, state.zoom);
    drawTriangle(ctx, ang.point3.x, ang.point3.y, triangleSize, angColor, state.zoom);
    
    // Draw lines
    ctx.strokeStyle = angColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
    
    ctx.beginPath();
    ctx.moveTo(ang.point1.x, ang.point1.y);
    ctx.lineTo(ang.point2.x, ang.point2.y);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(ang.point2.x, ang.point2.y);
    ctx.lineTo(ang.point3.x, ang.point3.y);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // --- Logic for Reflex Angles ---
    
    // Calculate the displayed angle value
    const displayedAngle = ang.isReflex ? (360 - ang.value) : ang.value;
    
    const vertex = ang.point2;
    const arcRadius = 20 / state.zoom;
    
    let angle1 = Math.atan2(ang.point1.y - vertex.y, ang.point1.x - vertex.x);
    let angle2 = Math.atan2(ang.point3.y - vertex.y, ang.point3.x - vertex.x);
    
    angle1 = ((angle1 % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
    angle2 = ((angle2 % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
    
    let diff = angle2 - angle1;
    if (diff < 0) diff += 2 * Math.PI;
    
    // Swap angles based on isReflex state
    if (ang.isReflex) {
      if (diff <= Math.PI) {
        [angle1, angle2] = [angle2, angle1];
      }
    } else {
      if (diff > Math.PI) {
        [angle1, angle2] = [angle2, angle1];
      }
    }
    
    ctx.strokeStyle = angColor;
    ctx.lineWidth = 2 / state.zoom;
    ctx.beginPath();
    ctx.arc(vertex.x, vertex.y, arcRadius, angle1, angle2, false);
    ctx.stroke();
    
    // --- Text Rendering with Click Box ---
    
    const textOffsetX = 35 / state.zoom;
    const textOffsetY = -35 / state.zoom;
    const textX = ang.point2.x + textOffsetX;
    const textY = ang.point2.y + textOffsetY;
    const angleText = `${Math.round(displayedAngle)}°`;
    
    // Measure for text box
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0); 
    ctx.font = `12px system-ui`;
    const metrics = ctx.measureText(angleText);
    const textWidthCanvas = metrics.width;
    const textHeightCanvas = 12; 
    ctx.restore();
    
    const canvasTextX = textX * state.zoom + state.panX;
    const canvasTextY = textY * state.zoom + state.panY;
    
    // Define Clickable Box (with padding)
    const padding = 10; 
    ang.textBox = {
      x: canvasTextX - padding,
      y: canvasTextY - textHeightCanvas - padding,
      width: textWidthCanvas + (padding * 2),
      height: textHeightCanvas + (padding * 2) 
    };
    
    // Draw text (Sky Blue)
    ctx.fillStyle = "#87CEEB"; 
    ctx.font = `${12 / state.zoom}px system-ui`;
    ctx.fillText(angleText, textX, textY);
    
    // Draw underline
    ctx.strokeStyle = "#87CEEB";
    ctx.lineWidth = 1 / state.zoom;
    ctx.beginPath();
    ctx.moveTo(textX, textY + 2 / state.zoom);
    ctx.lineTo(textX + textWidthCanvas / state.zoom, textY + 2 / state.zoom);
    ctx.stroke();
  }

  ctx.restore();
}

function onCanvasMouseDown(evt) {
  const rect = canvas.getBoundingClientRect();
  const canvasX = evt.clientX - rect.left;
  const canvasY = evt.clientY - rect.top;
  const world = canvasToWorld(canvasX, canvasY);

  if (state.mode === "idle") {

    for (const ang of state.measurements.angles) {
      if (ang.textBox && isPointInRect(canvasX, canvasY, ang.textBox)) {
        console.log("Angle text clicked:", ang.id, "toggling from", ang.isReflex, "to", !ang.isReflex);
        ang.isReflex = !ang.isReflex;
        updateLists();
        render();
        // Auto-save immediately so the change persists
        saveCurrentFrameAnnotations(); 
        evt.preventDefault();
        evt.stopPropagation();
        return; 
      }
    }
    // Check for draggable elements first
    const kp = findNearestKeypoint(world.x, world.y, 15);
    if (kp) {
      state.dragging = kp;
      state.dragStartX = world.x;
      state.dragStartY = world.y;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    const lineEndpoint = findNearestLineEndpoint(world.x, world.y, 15);
    if (lineEndpoint) {
      state.draggingLineEndpoint = lineEndpoint;
      state.dragStartX = world.x;
      state.dragStartY = world.y;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    const distanceEndpoint = findNearestDistanceEndpoint(world.x, world.y, 15);
    if (distanceEndpoint) {
      state.draggingDistanceEndpoint = distanceEndpoint;
      state.dragStartX = world.x;
      state.dragStartY = world.y;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    const anglePoint = findNearestAnglePoint(world.x, world.y, 15);
    if (anglePoint) {
      state.draggingAnglePoint = anglePoint;
      state.dragStartX = world.x;
      state.dragStartY = world.y;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    const roiPoint = findNearestRoiPoint(world.x, world.y, 15);
    if (roiPoint) {
      state.draggingRoiPoint = roiPoint;
      state.dragStartX = world.x;
      state.dragStartY = world.y;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    
    // Check for bounding box click (check this BEFORE panning to allow selection)
    const clickedTrack = findBoundingBoxAt(world.x, world.y);
    if (clickedTrack) {
      const trackId = clickedTrack.track_id || clickedTrack.id;
      state.selectedTrackId = trackId;
      updateBboxInfo();
      render();
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }
    
    // If not clicking on anything, allow panning
    if (evt.button === 0 && !evt.shiftKey) {
      state.isPanning = true;
      state.panStartX = canvasX - state.panX;
      state.panStartY = canvasY - state.panY;
      canvas.style.cursor = "grabbing";
      evt.preventDefault();
      return;
    }
  }

  if (evt.button === 1 || (evt.button === 0 && evt.shiftKey)) {
    state.isPanning = true;
    state.panStartX = canvasX - state.panX;
    state.panStartY = canvasY - state.panY;
    canvas.style.cursor = "grabbing";
    evt.preventDefault();
    return;
  }

  onCanvasClick(evt);
}

function onCanvasMouseMove(evt) {
  const rect = canvas.getBoundingClientRect();
  const canvasX = evt.clientX - rect.left;
  const canvasY = evt.clientY - rect.top;
  const world = canvasToWorld(canvasX, canvasY);

  if (state.dragging) {
    state.dragging.x = world.x;
    state.dragging.y = world.y;
    updateLists();
    render();
    return;
  }

  if (state.draggingLineEndpoint) {
    const { line, endpoint } = state.draggingLineEndpoint;
    if (endpoint === "start") {
      line.start.x = world.x;
      line.start.y = world.y;
    } else {
      line.end.x = world.x;
      line.end.y = world.y;
    }
    updateLists();
    render();
    return;
  }

  if (state.draggingDistanceEndpoint) {
    const { distance, endpoint } = state.draggingDistanceEndpoint;
    if (endpoint === "start") {
      distance.start.x = world.x;
      distance.start.y = world.y;
    } else {
      distance.end.x = world.x;
      distance.end.y = world.y;
    }
    distance.value = Math.hypot(distance.end.x - distance.start.x, distance.end.y - distance.start.y);
    updateLists();
    render();
    return;
  }

  if (state.draggingAnglePoint) {
    const { angle, pointType } = state.draggingAnglePoint;
    angle[pointType].x = world.x;
    angle[pointType].y = world.y;
    angle.value = angleFromThreePoints(angle.point1, angle.point2, angle.point3);
    updateLists();
    render();
    return;
  }

  if (state.draggingRoiPoint) {
    const { roi, pointIndex } = state.draggingRoiPoint;
    roi.points[pointIndex].x = world.x;
    roi.points[pointIndex].y = world.y;
    updateLists();
    render();
    return;
  }

  if (state.isPanning) {
    state.panX = canvasX - state.panStartX;
    state.panY = canvasY - state.panStartY;
    render();
    return;
  }

  if ((state.mode === "draw-line" || state.mode === "distance" || state.mode === "angle" || state.mode === "roi") && 
      state.tempPoints.length >= 1) {
    state.previewMouseX = world.x;
    state.previewMouseY = world.y;
    render();
  } else {
    state.previewMouseX = null;
    state.previewMouseY = null;
  }

  // Update cursor
  if (state.mode === "idle") {

    // Check if hovering over angle text
    const rect = canvas.getBoundingClientRect();
    const cvsX = evt.clientX - rect.left;
    const cvsY = evt.clientY - rect.top;
    let hoveringAngleText = false;
    for (const ang of state.measurements.angles) {
      if (ang.textBox && isPointInRect(cvsX, cvsY, ang.textBox)) {
        hoveringAngleText = true;
        break;
      }
    }

    const kp = findNearestKeypoint(world.x, world.y, 15);
    const lineEndpoint = findNearestLineEndpoint(world.x, world.y, 15);
    const distanceEndpoint = findNearestDistanceEndpoint(world.x, world.y, 15);
    const anglePoint = findNearestAnglePoint(world.x, world.y, 15);
    const roiPoint = findNearestRoiPoint(world.x, world.y, 15);
    const clickedTrack = findBoundingBoxAt(world.x, world.y);



    if (kp || lineEndpoint || distanceEndpoint || anglePoint || roiPoint || clickedTrack || hoveringAngleText) {
      canvas.style.cursor = "pointer";
    } else {
      canvas.style.cursor = "move";
    }
  } else if (state.mode === "add-keypoint") {
    const kp = findNearestKeypoint(world.x, world.y, 15);
    canvas.style.cursor = kp ? "pointer" : "crosshair";
  } else if (state.mode === "draw-line" || state.mode === "distance" || state.mode === "angle" || state.mode === "roi") {
    canvas.style.cursor = "crosshair";
  } else {
    canvas.style.cursor = "default";
  }
}

function onCanvasMouseUp(evt) {
  if (state.dragging) {
    state.dragging = null;
    canvas.style.cursor = "default";
  }
  if (state.draggingLineEndpoint) {
    state.draggingLineEndpoint = null;
    canvas.style.cursor = "default";
  }
  if (state.draggingDistanceEndpoint) {
    state.draggingDistanceEndpoint = null;
    canvas.style.cursor = "default";
  }
  if (state.draggingAnglePoint) {
    state.draggingAnglePoint = null;
    canvas.style.cursor = "default";
  }
  if (state.draggingRoiPoint) {
    state.draggingRoiPoint = null;
    canvas.style.cursor = "default";
  }
  if (state.isPanning) {
    state.isPanning = false;
    canvas.style.cursor = "default";
  }
}

function onCanvasMouseLeave(evt) {
  state.previewMouseX = null;
  state.previewMouseY = null;
  if (state.mode === "draw-line" && state.tempPoints.length === 1) {
    render();
  }
}

function onCanvasWheel(evt) {
  evt.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const canvasX = evt.clientX - rect.left;
  const canvasY = evt.clientY - rect.top;
  
  let delta = evt.deltaY;
  const isTrackpad = evt.deltaMode === 0;
  
  if (isTrackpad) {
    const hasBothDeltas = Math.abs(evt.deltaX) > 0 && Math.abs(evt.deltaY) > 0;
    const isPinch = evt.ctrlKey || hasBothDeltas;
    
    if (isPinch) {
      let pinchDelta;
      if (Math.abs(evt.deltaX) > Math.abs(evt.deltaY)) {
        pinchDelta = evt.deltaX * 0.5;
      } else {
        pinchDelta = evt.deltaY * 0.5;
      }
      delta = -pinchDelta;
    } else {
      delta = evt.deltaY * 0.5;
    }
  } else if (evt.ctrlKey) {
    delta = evt.deltaY * 0.5;
  }
  
  if (Math.abs(delta) > 0.1) {
    zoomAtPoint(canvasX, canvasY, delta);
  }
}

async function onCanvasClick(evt) {
  const rect = canvas.getBoundingClientRect();
  const canvasX = evt.clientX - rect.left;
  const canvasY = evt.clientY - rect.top;
  const world = canvasToWorld(canvasX, canvasY);

  if (state.mode === "add-keypoint") {
    await addKeypoint(world.x, world.y);
    setMode("idle", "Keypoint added.");
    return;
  }

  if (state.mode === "draw-line") {
    const kp = findNearestKeypoint(world.x, world.y, 12);
    state.tempPoints.push({ x: kp ? kp.x : world.x, y: kp ? kp.y : world.y, keypointId: kp?.id });
    if (state.tempPoints.length === 1) {
      render();
    }
    if (state.tempPoints.length === 2) {
      const [start, end] = state.tempPoints;
      const label = `#${state.nextLineIndex}`;
      const color = COLOR_OPTIONS[lineColorIndex % COLOR_OPTIONS.length];
      lineColorIndex += 1;
      const id = `line-${state.nextLineIndex}`;
      const line = { id, label, start, end, color };
      
      // Associate with selected bounding box if one is selected
      if (state.selectedTrackId) {
        if (!state.boundingBoxes[state.selectedTrackId]) {
          state.boundingBoxes[state.selectedTrackId] = {};
        }
        if (!state.boundingBoxes[state.selectedTrackId].annotations) {
          state.boundingBoxes[state.selectedTrackId].annotations = { keypoints: [], lines: [], rois: [], measurements: { distances: [], angles: [] } };
        }
        state.boundingBoxes[state.selectedTrackId].annotations.lines.push(line);
      }
      
      state.lines.push(line);
      state.nextLineIndex += 1;
      state.tempPoints = [];
      updateLists();
      render();
      setMode("idle", "Line added.");
      
      // Auto-save annotations
      await saveCurrentFrameAnnotations();
    } else {
      setMode("draw-line", "Select end point for the line.");
    }
    return;
  }

  if (state.mode === "roi") {
    const kp = findNearestKeypoint(world.x, world.y, 12);
    state.tempPoints.push({ x: kp ? kp.x : world.x, y: kp ? kp.y : world.y, keypointId: kp?.id });
    render();
    const pointCount = state.tempPoints.length;
    if (pointCount < 3) {
      setMode("roi", `Click at least ${3 - pointCount} more point(s) to create ROI ${state.nextRoiIndex}. Press Enter to complete.`);
    } else {
      setMode("roi", `ROI has ${pointCount} points. Press Enter to complete ROI ${state.nextRoiIndex}.`);
    }
    return;
  }

  if (state.mode === "distance") {
    const kp = findNearestKeypoint(world.x, world.y, 12);
    state.tempPoints.push({ x: kp ? kp.x : world.x, y: kp ? kp.y : world.y, keypointId: kp?.id });
    if (state.tempPoints.length === 1) {
      render();
    }
    if (state.tempPoints.length === 2) {
      const [start, end] = state.tempPoints;
      const dist = Math.hypot(start.x - end.x, start.y - end.y);
      const label = `#${state.nextDistanceIndex}`;
      const color = COLOR_OPTIONS[distanceColorIndex % COLOR_OPTIONS.length];
      distanceColorIndex += 1;
      const distance = {
        id: `dist-${state.nextDistanceIndex}`,
        label,
        start: { x: start.x, y: start.y },
        end: { x: end.x, y: end.y },
        value: dist,
        color,
      };
      
      // Associate with selected bounding box if one is selected
      if (state.selectedTrackId) {
        if (!state.boundingBoxes[state.selectedTrackId]) {
          state.boundingBoxes[state.selectedTrackId] = {};
        }
        if (!state.boundingBoxes[state.selectedTrackId].annotations) {
          state.boundingBoxes[state.selectedTrackId].annotations = { keypoints: [], lines: [], rois: [], measurements: { distances: [], angles: [] } };
        }
        state.boundingBoxes[state.selectedTrackId].annotations.measurements.distances.push(distance);
      }
      
      state.measurements.distances.push(distance);
      state.nextDistanceIndex += 1;
      state.tempPoints = [];
      updateLists();
      render();
      setMode("idle", `Distance: ${dist.toFixed(2)} px`);
      
      // Auto-save annotations
      await saveCurrentFrameAnnotations();
    } else {
      setMode("distance", "Select second point.");
    }
    return;
  }

  if (state.mode === "angle") {
    const kp = findNearestKeypoint(world.x, world.y, 12);
    state.tempPoints.push({ x: kp ? kp.x : world.x, y: kp ? kp.y : world.y, keypointId: kp?.id });
    if (state.tempPoints.length === 1) {
      setMode("angle", "Select second point (vertex).");
      render();
    } else if (state.tempPoints.length === 2) {
      setMode("angle", "Select third point.");
      render();
    } else if (state.tempPoints.length === 3) {
      const [p1, p2, p3] = state.tempPoints;
      const ang = angleFromThreePoints(p1, p2, p3);
      const label = `#${state.nextAngleIndex}`;
      const color = COLOR_OPTIONS[angleColorIndex % COLOR_OPTIONS.length];
      angleColorIndex += 1;
      const angle = {
        id: `angle-${state.nextAngleIndex}`,
        label,
        point1: { x: p1.x, y: p1.y },
        point2: { x: p2.x, y: p2.y },
        point3: { x: p3.x, y: p3.y },
        value: ang,
        isReflex: false, // ADD THIS LINE
        color,
      };
      
      // Associate with selected bounding box if one is selected
      if (state.selectedTrackId) {
        if (!state.boundingBoxes[state.selectedTrackId]) {
          state.boundingBoxes[state.selectedTrackId] = {};
        }
        if (!state.boundingBoxes[state.selectedTrackId].annotations) {
          state.boundingBoxes[state.selectedTrackId].annotations = { keypoints: [], lines: [], rois: [], measurements: { distances: [], angles: [] } };
        }
        state.boundingBoxes[state.selectedTrackId].annotations.measurements.angles.push(angle);
      }
      
      state.measurements.angles.push(angle);
      state.nextAngleIndex += 1;
      state.tempPoints = [];
      updateLists();
      render();
      setMode("idle", `Angle: ${ang.toFixed(2)}°`);
      
      // Auto-save annotations
      await saveCurrentFrameAnnotations();
    }
    return;
  }
}

async function saveCurrentFrameAnnotations() {
  const studyIdToUse = state.studyId || state.tempStudyId;
  if (state.currentFrameIndex === null || !studyIdToUse) {
    return;
  }
  
  // Capture the frame index at the start to prevent race conditions
  const frameIndexToSave = state.currentFrameIndex;
  
  // Create frame-specific bounding boxes object
  // Only include bounding boxes that have labels (name or action)
  const frameBoundingBoxes = {};
  for (const [trackId, bboxData] of Object.entries(state.boundingBoxes)) {
    if (bboxData.name || bboxData.action) {
      frameBoundingBoxes[trackId] = {
        name: bboxData.name || null,
        action: bboxData.action || null,
        annotations: bboxData.annotations || null,
      };
    }
  }
  
  const annotations = {
    keypoints: state.keypoints || [],
    lines: state.lines || [],
    rois: state.rois || [],
    measurements: state.measurements || { distances: [], angles: [] },
    bounding_boxes: frameBoundingBoxes,
  };
  
  // Log what we're saving
  console.log(`[saveCurrentFrameAnnotations] Saving frame ${frameIndexToSave}: keypoints=${annotations.keypoints.length}, lines=${annotations.lines.length}, rois=${annotations.rois.length}, distances=${annotations.measurements.distances.length}, angles=${annotations.measurements.angles.length}`);
  
  try {
    const url = `/study/${encodeURIComponent(studyIdToUse)}/frame/${frameIndexToSave}/annotations`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotations),
    });
    if (!res.ok) {
      console.error(`[saveCurrentFrameAnnotations] Failed to save: ${res.status} - ${await res.text()}`);
    }
  } catch (error) {
    console.error("Error saving frame annotations:", error);
  }
}

async function applyLabels() {
  if (!state.selectedTrackId) {
    alert("Please select a bounding box first.");
    return;
  }
  
  const name = playerNameInput.value.trim();
  const action = actionInput.value.trim();
  
  if (!name && !action) {
    alert("Please enter at least a name or action.");
    return;
  }
  
  // Update bounding box data for current frame (for immediate visual feedback)
  if (!state.boundingBoxes[state.selectedTrackId]) {
    state.boundingBoxes[state.selectedTrackId] = {};
  }
  
  if (name) {
    state.boundingBoxes[state.selectedTrackId].name = name;
  } else {
    // Clear name if empty
    delete state.boundingBoxes[state.selectedTrackId].name;
  }
  
  if (action) {
    state.boundingBoxes[state.selectedTrackId].action = action;
  } else {
    // Clear action if empty
    delete state.boundingBoxes[state.selectedTrackId].action;
  }
  
  // Save current frame annotations
  await saveCurrentFrameAnnotations();
  
  // Propagate to all frames with the same track_id
  await propagateLabels(true);
  
  // Reload current frame annotations to ensure consistency
  await loadFrameAnnotations(state.currentFrameIndex);
  
  updateBboxInfo();
  render();
}

async function propagateLabels(silent = false) {
  if (!state.selectedTrackId) {
    if (!silent) {
      alert("Please select a bounding box first.");
    }
    return;
  }
  
  const studyIdToUse = state.studyId || state.tempStudyId;
  if (!studyIdToUse) {
    // Auto-create temp study ID if not set
    if (!state.tempStudyId) {
      state.tempStudyId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
  }
  
  const name = playerNameInput.value.trim();
  const action = actionInput.value.trim();
  
  if (!name && !action) {
    // If both are empty, clear labels from all frames
    // This allows clearing labels by deleting the text
  }
  
  // Save current frame annotations first
  await saveCurrentFrameAnnotations();
  
  // Propagate labels to all frames with this track_id
  try {
    const studyIdToUseFinal = state.studyId || state.tempStudyId;
    const res = await fetch(`/study/${encodeURIComponent(studyIdToUseFinal)}/propagate_labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_id: state.selectedTrackId,
        frame_index: state.currentFrameIndex,
        name: name || null,
        action: action || null,
        video_id: state.videoId || null,  // Include video_id for temp studies
      }),
    });
    
    if (res.ok) {
      const data = await res.json();
      if (!silent) {
        console.log(`Labels propagated to ${data.count} frames for track ${state.selectedTrackId}`);
      }
      
      // Reload current frame to get updated bounding boxes
      await loadFrameAnnotations(state.currentFrameIndex);
      updateBboxInfo();
      render();
    } else {
      if (!silent) {
        alert(`Failed to propagate labels: ${await res.text()}`);
      } else {
        console.error(`Failed to propagate labels: ${await res.text()}`);
      }
    }
  } catch (error) {
    console.error("Error propagating labels:", error);
    if (!silent) {
      alert(`Failed to propagate labels: ${error.message}`);
    }
  }
}

async function saveStudy() {
  const studyId = studyIdInput.value.trim();
  if (!studyId) {
    alert("Enter a study ID.");
    return;
  }
  if (!state.imageUrl) {
    alert("Upload a file first.");
    return;
  }
  
  // CRITICAL: Save current frame annotations FIRST (under the current study ID)
  // This ensures any annotations created on the current frame are persisted before migration
  const oldTempStudyId = state.tempStudyId;
  const oldStudyId = state.studyId;
  const currentStudyIdForSave = oldStudyId || oldTempStudyId;
  
  console.log(`[saveStudy] Saving current frame ${state.currentFrameIndex} annotations under ${currentStudyIdForSave}`);
  await saveCurrentFrameAnnotations();
  
  // Small delay to ensure save completes
  await new Promise(resolve => setTimeout(resolve, 100));
  
  // If we have a temp study ID with annotations, migrate them to the new study ID
  // This ensures annotations created under the temp ID are preserved when saving
  if (oldTempStudyId && oldTempStudyId !== studyId) {
    console.log(`[saveStudy] Migrating annotations from temp study ${oldTempStudyId} to ${studyId}`);
    try {
      // Migrate annotations from temp study to new study ID
      const tempRes = await fetch(`/study/${encodeURIComponent(oldTempStudyId)}/migrate/${encodeURIComponent(studyId)}`, {
        method: "POST",
      });
      if (!tempRes.ok) {
        const errorText = await tempRes.text();
        console.warn(`[saveStudy] Migration failed, but continuing: ${errorText}`);
      } else {
        const result = await tempRes.json();
        console.log(`[saveStudy] Successfully migrated ${result.frames_migrated || 0} frames from ${oldTempStudyId} to ${studyId}`);
      }
    } catch (error) {
      console.warn(`[saveStudy] Migration error (continuing anyway): ${error.message}`);
    }
  }
  
  // Also migrate from old study ID if different from new
  if (oldStudyId && oldStudyId !== studyId && oldStudyId !== oldTempStudyId) {
    console.log(`[saveStudy] Migrating annotations from old study ${oldStudyId} to ${studyId}`);
    try {
      const oldRes = await fetch(`/study/${encodeURIComponent(oldStudyId)}/migrate/${encodeURIComponent(studyId)}`, {
        method: "POST",
      });
      if (oldRes.ok) {
        const result = await oldRes.json();
        console.log(`[saveStudy] Successfully migrated ${result.frames_migrated || 0} frames from ${oldStudyId} to ${studyId}`);
      }
    } catch (error) {
      console.warn(`[saveStudy] Migration from old study error: ${error.message}`);
    }
  }
  
  // Now update study IDs - this ensures future saves go to the right place
  state.studyId = studyId;
  state.tempStudyId = studyId; // Update temp study ID to match so annotations persist after save
  
  // Create payload (no image saving)
  const payload = {
    study_id: studyId,
    image_url: state.imageUrl,
    stored_filename: state.storedFilename,
    original_filename: state.originalFilename,
    kind: state.kind,
    metadata: state.metadata,
    video_id: state.videoId,
    frames: state.frames,
    total_frames: state.frames.length,
  };
  
  const formData = new FormData();
  formData.append("payload", JSON.stringify(payload));
  
  const res = await fetch(`/study/${encodeURIComponent(studyId)}/save`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    alert(`Save failed: ${await res.text()}`);
    return;
  }
  
  // After successful save, reload current frame annotations to ensure they persist
  // This ensures annotations are visible after saving
  console.log(`[saveStudy] Reloading frame ${state.currentFrameIndex} annotations after save`);
  await loadFrameAnnotations(state.currentFrameIndex);
  render();
  
  alert(`Study "${studyId}" has been saved successfully.`);
  setMode("idle", `Saved study ${studyId}`);
}

async function showStudySelectionModal() {
  const res = await fetch("/studies");
  if (!res.ok) {
    alert("Failed to load study list.");
    return;
  }
  const data = await res.json();
  const studies = data.studies || [];
  
  if (studies.length === 0) {
    studyListContainer.innerHTML = "<p>No saved studies found.</p>";
    studyModal.style.display = "block";
    return;
  }
  
  studyListContainer.innerHTML = "";
  studies.forEach((study) => {
    const studyItem = document.createElement("div");
    studyItem.className = "study-item";
    studyItem.setAttribute("role", "button");
    studyItem.setAttribute("tabindex", "0");
    studyItem.setAttribute("data-study-id", study);
    
    const studyId = document.createElement("div");
    studyId.className = "study-id";
    studyId.textContent = study;
    
    studyItem.appendChild(studyId);
    
    const clickHandler = () => {
      loadStudyById(study);
      studyModal.style.display = "none";
    };
    
    studyItem.addEventListener("click", clickHandler);
    studyItem.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        clickHandler();
      }
    });
    
    studyListContainer.appendChild(studyItem);
  });
  
  studyModal.style.display = "block";
}

async function loadStudyById(studyId) {
  if (!studyId) {
    alert("No study ID provided.");
    return;
  }
  
  setMode("loading", "Loading study...");

  // Reset server session memory and clear temp files to ensure fresh state
  // This prevents old temp videos from accumulating
  try {
    await fetch("/reset_session?clear_temp=true", { method: "POST" });
    console.log("[loadStudyById] Session reset and temp files cleared");
  } catch (err) {
    console.warn("Failed to reset session memory:", err);
  }

  const res = await fetch(`/study/${encodeURIComponent(studyId)}`);
  if (!res.ok) {
    alert(`Load failed: ${await res.text()}`);
    setMode("idle", "Load failed");
    return;
  }
  
  const data = await res.json();
  
  console.log(`[loadStudyById] Loaded study data:`, {
    study_id: data.study_id,
    video_id: data.video_id,
    total_frames: data.total_frames,
    has_keypoints: data.metadata?.has_keypoints,
    keypoints_tracks_frames: Object.keys(data.keypoints_tracks || {}).length
  });
  
  studyIdInput.value = studyId;
  state.studyId = studyId;
  state.tempStudyId = studyId; // Use same ID for temp to ensure annotations are found
  
  // Load video data
  state.videoId = data.video_id;
  state.storedFilename = `video_${data.video_id}.mp4`; // Video is now in temp
  state.originalFilename = data.original_filename;
  state.frames = data.frames || [];
  state.hasKeypointsFile = data.metadata?.has_keypoints || false;
  state.metadata = data.metadata || {};
  state.kind = data.kind || "video";
  state.imageUrl = `/video/${data.video_id}/frame/0`;
  
  // Normalize keypointsTracks keys (handle both string and number keys)
  const rawTracks = data.keypoints_tracks || {};
  state.keypointsTracks = {};
  if (rawTracks && typeof rawTracks === 'object' && Object.keys(rawTracks).length > 0) {
    for (const key in rawTracks) {
      if (rawTracks.hasOwnProperty(key)) {
        const numKey = parseInt(key, 10);
        const value = rawTracks[key];
        if (!isNaN(numKey) && Array.isArray(value)) {
          state.keypointsTracks[numKey] = value;
          state.keypointsTracks[String(numKey)] = value;
        }
      }
    }
    console.log(`[loadStudyById] Loaded ${Object.keys(state.keypointsTracks).length / 2} frames with keypoint tracks`);
  }
  
  // Reset annotation state for fresh load
  state.currentFrameIndex = 0;
  state.boundingBoxes = {};
  state.selectedTrackId = null;
  state.keypoints = [];
  state.lines = [];
  state.rois = [];
  state.measurements = { distances: [], angles: [] };
  state.nextKeypointIndex = 1;
  state.nextLineIndex = 1;
  state.nextRoiIndex = 1;
  state.nextDistanceIndex = 1;
  state.nextAngleIndex = 1;
  
  metadataOutput.textContent = JSON.stringify(state.metadata, null, 2);
  
  if (frameNavigationInline) frameNavigationInline.style.display = "inline-flex";
  
  // Setup frame number editor
  setupFrameNumberEditor();
  
  // Load first frame (this will also load annotations from temp file)
  await loadFrame(0);
  setMode("idle", `Loaded study ${studyId}`);
}

async function loadStudy() {
  await showStudySelectionModal();
}

// Event listeners
canvas.addEventListener("mousedown", onCanvasMouseDown);
canvas.addEventListener("mousemove", onCanvasMouseMove);
canvas.addEventListener("mouseup", onCanvasMouseUp);
canvas.addEventListener("mouseleave", (evt) => {
  onCanvasMouseUp(evt);
  onCanvasMouseLeave(evt);
});
canvas.addEventListener("wheel", onCanvasWheel);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// Handle file selection (just show feedback, don't upload)
videoInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    videoFileInfo.textContent = `Selected: ${file.name} (${sizeMB} MB)`;
    videoFileInfo.style.color = "#4caf50";
    updateUploadButtonState();
  } else {
    videoFileInfo.textContent = "";
    updateUploadButtonState();
  }
});

keypointsInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    keypointsFileInfo.textContent = `Selected: ${file.name} (${sizeMB} MB)`;
    keypointsFileInfo.style.color = "#4caf50";
    console.log(`[keypointsInput] File selected: ${file.name}, size: ${file.size} bytes`);
  } else {
    keypointsFileInfo.textContent = "";
    console.log(`[keypointsInput] File selection cleared`);
  }
  updateUploadButtonState();
});

// Upload button click handler
console.log("[init] Upload button element:", uploadBtn);
if (uploadBtn) {
  uploadBtn.addEventListener("click", (e) => {
    console.log("[uploadBtn] Click event fired");
    e.preventDefault();
    e.stopPropagation();
    uploadVideo();
  });
  console.log("[init] Upload button click handler attached");
} else {
  console.error("[uploadBtn] Upload button not found!");
}

function updateUploadButtonState() {
  if (!uploadBtn) return;
  const uploadBtnText = document.getElementById("upload-btn-text");
  const hasVideo = videoInput && videoInput.files.length > 0;
  if (hasVideo) {
    uploadBtn.disabled = false;
    uploadBtn.style.opacity = "1";
    uploadBtn.style.cursor = "pointer";
    uploadBtn.style.backgroundColor = "#4caf50";
    if (uploadBtnText) {
      uploadBtnText.textContent = "Upload";
    }
  } else {
    uploadBtn.disabled = true;
    uploadBtn.style.opacity = "0.6";
    uploadBtn.style.cursor = "not-allowed";
    uploadBtn.style.backgroundColor = "#cccccc";
    if (uploadBtnText) {
      uploadBtnText.textContent = "Upload";
    }
  }
}

// Add hover effect
if (uploadBtn) {
  uploadBtn.addEventListener("mouseenter", () => {
    if (!uploadBtn.disabled) {
      uploadBtn.style.backgroundColor = "#45a049";
    }
  });

  uploadBtn.addEventListener("mouseleave", () => {
    if (!uploadBtn.disabled) {
      uploadBtn.style.backgroundColor = "#4caf50";
    }
  });
  
  // Ensure progress bar is initially hidden
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  if (uploadProgressBar) {
    uploadProgressBar.style.width = "0%";
  }
}

// Initialize button state
updateUploadButtonState();

createKeypointBtn.addEventListener("click", startAddKeypointMode);
drawLineBtn.addEventListener("click", startDrawLineMode);
distanceBtn.addEventListener("click", startDistanceMode);
angleBtn.addEventListener("click", startAngleMode);
roiBtn.addEventListener("click", startRoiMode);
saveStudyBtn.addEventListener("click", saveStudy);
loadStudyBtn.addEventListener("click", loadStudy);
applyLabelsBtn.addEventListener("click", applyLabels);

// Handle Enter key to complete ROI
document.addEventListener("keydown", async (evt) => {
  if ((evt.key === "Enter" || evt.key === "Return") && state.mode === "roi" && state.tempPoints.length >= 3) {
    evt.preventDefault();
    await completeRoi();
  }
  
  // Frame navigation
  if (evt.key === "ArrowLeft" && state.currentFrameIndex > 0) {
    evt.preventDefault();
    loadFrame(state.currentFrameIndex - 1);
  } else if (evt.key === "ArrowRight" && state.currentFrameIndex < state.frames.length - 1) {
    evt.preventDefault();
    loadFrame(state.currentFrameIndex + 1);
  }
});

// Modal event listeners
if (closeModalBtn) {
  closeModalBtn.addEventListener("click", () => {
    studyModal.style.display = "none";
  });
}

if (studyModal) {
  studyModal.addEventListener("click", (event) => {
    if (event.target === studyModal) {
      studyModal.style.display = "none";
    }
  });
}

document.addEventListener("keydown", (evt) => {
  if (evt.key === "Escape" && studyModal && studyModal.style.display === "block") {
    studyModal.style.display = "none";
  }
});

zoomInBtn.addEventListener("click", () => {
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  zoomAtPoint(centerX, centerY, 1);
});

zoomOutBtn.addEventListener("click", () => {
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  zoomAtPoint(centerX, centerY, -1);
});

resetViewBtn.addEventListener("click", () => {
  resetView();
  render();
});

const PAN_STEP = 40;
panUpBtn?.addEventListener("click", () => panBy(0, PAN_STEP));
panDownBtn?.addEventListener("click", () => panBy(0, -PAN_STEP));
panLeftBtn?.addEventListener("click", () => panBy(PAN_STEP, 0));
panRightBtn?.addEventListener("click", () => panBy(-PAN_STEP, 0));

// Frame navigation event listeners
prevFrameBtn?.addEventListener("click", () => {
  if (state.currentFrameIndex > 0) {
    loadFrame(state.currentFrameIndex - 1);
  }
});

nextFrameBtn?.addEventListener("click", () => {
  if (state.currentFrameIndex < state.frames.length - 1) {
    loadFrame(state.currentFrameIndex + 1);
  }
});

function isPointInRect(x, y, rect) {
  return x >= rect.x && x <= rect.x + rect.width &&
         y >= rect.y && y <= rect.y + rect.height;
}

// Initialize
fitCanvas();
setMode("idle", "Upload a video file to begin.");
