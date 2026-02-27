# DIY Soccer Broadcast System — Design Document

## Project Goal

Recreate a Veo-style AI soccer camera system on the cheap. Capture an entire soccer pitch with two wide-angle cameras, stitch them into one panoramic video, then use an interactive Python app to create a broadcast-quality 1080p output with virtual pan/zoom and a scoreboard overlay. Replay clips are added afterward in Clipchamp.

---

## Hardware Setup

### Cameras
- **2x GoPro Hero 13** at 5.3K resolution (5312×2988) with standard wide lens (~155° FOV)
- Alternatively: 4K (3840×2160) with **Ultra Wide Lens Mod** (177° FOV) for maximum pitch coverage
- Both cameras set to identical manual settings: resolution, frame rate, white balance, color profile, exposure
- Lock everything to manual — auto settings can drift independently between cameras and create visible seam differences

### Mount
- **Off-the-shelf dual GoPro mount adapter** (~$10-20 on Amazon) — aluminum, holds two cameras side by side
- Hero 13 has built-in 1/4"-20 thread on bottom, so a **dual camera bracket bar** (SmallRig-style) also works and gives more rigid alignment
- Cameras should be as close together as possible to minimize parallax at the stitch seam
- Mount goes on a tall tripod (10-13 feet) at midfield for elevated view of the full pitch

### Overlap
- With two cameras at ~155-177° FOV each, there will be significant overlap in the center (20-30%)
- The overlap region is where the stitch blends — more overlap = cleaner seam
- Although the mount is rigid, small differences in tripod placement and camera tightening between games mean stitch calibration should be run **per-game** (automated, ~10-15 seconds)

### Resolution Math
- Two 5.3K streams (5312px wide each) with ~25% overlap → stitched output ~**8000 × 2988 pixels**
- Panorama aspect ratio: **2.68:1** (much wider than 16:9 / 1.78:1)

### Zoom Range

Output is always 1920×1080. Zoom controls the size of the crop window taken from the panorama. The crop is always downscaled or near-native — never more than 20% upscaled, so there's no visible quality loss.

**Zoom = 1.0x (baseline):** Crop = 1920×1080 (1:1 pixel mapping, native quality)
- Horizontal pan range: 8000 − 1920 = 6,080 pixels of travel
- Vertical pan range: 2988 − 1080 = 1,908 pixels of travel

**Max zoom in = 1.2x** (20% upscale — imperceptible in motion video)
- Crop = 1600 × 900 pixels, upscaled to 1920×1080
- Horizontal pan range: 8000 − 1600 = 6,400 pixels (~5x horizontal travel)
- On a youth field (~80 yards), frames roughly a 16-yard wide window at max zoom
- Still plenty of source pixels — no visible softness on TV or YouTube

**Max zoom out = full panorama (letterboxed)**
- Shows the entire 8000×2988 panorama scaled to fit within 1920×1080
- Panorama scaled to 1920px wide → height = 2988 × (1920 ÷ 8000) = **717 pixels**
- Centered in 1080p frame with **~182px black bars** top and bottom
- Useful for kickoffs, set pieces, or showing full-field context

---

## Recording Workflow

1. Mount both GoPros in the fixed dual bracket on the tripod at midfield
2. Open GoPro Quik app and **sync timecode** across both cameras — this tags every frame with matching timestamps
3. Start both cameras via the Quik app or manually
4. Leave cameras running for the full match (30-min halves + halftime + buffer ≈ 85 minutes)
5. Stop recording, pull SD cards

*If you forget step 2, the software will auto-detect missing timecode and fall back to audio sync.*

---

## Software Pipeline Overview

The system has 3 stages, 2 automated and 1 interactive:

```
Stage 1: Calibrate + Stitch  (automated, ~15-20 minutes)
Stage 2: Interactive PTZ     (interactive, ~match length — you watch once)
Stage 3: Render              (automated, ~30-60 minutes)
```

Optional Stage 4: Add replay clips in Clipchamp (manual, ~20 minutes)

Frame sync between cameras is handled by **GoPro timecode sync** (set up before recording via the Quik app). If timecode sync was forgotten, the system falls back to **audio cross-correlation** automatically.

All processing runs locally on an HP Spectre laptop (Intel Iris Xe integrated graphics). The key design decision is **splitting the interactive session from the render** so the laptop never has to decode + encode simultaneously.

---

## Stage 1: Calibrate + Stitch

**Purpose:** Compute per-game stitch calibration, then combine the two time-synced GoPro videos into a single wide panoramic video.

### Timecode Sync (handled at recording time)
The Hero 13's timecode sync feature (via GoPro Quik) tags every frame with matching timestamps. In post, read the timecode from each file's metadata via ffprobe to determine the frame offset — this is a one-liner, no audio correlation needed.

### Fallback: Audio Sync
If timecode sync was forgotten at the field, fall back to audio cross-correlation:
1. Extract audio from both video files via ffmpeg (`-vn -ac 1 -ar 16000 -f wav`)
2. Cross-correlate the two waveforms with numpy/scipy to find the time offset
3. Ambient crowd/whistle noise is enough for sub-frame accuracy — no clap needed, but a clap helps

The Streamlit UI (Page 1) auto-detects whether timecode metadata is present. If found, uses timecode. If not, runs audio sync automatically.

### Per-Game Calibration (~10-15 seconds)
Even with a rigid mount, small variations in setup between games (tripod angle, camera tightening, bracket flex) can shift the overlap geometry enough to break a saved homography. Calibration runs **automatically at the start of each game's processing:**

1. Extract a single frame from each camera
2. Use OpenCV to find matching feature points (SIFT/ORB) in the overlap region
3. Compute a homography matrix with RANSAC
4. Save to a per-game calibration file

Soccer field markings (center circle, halfway line, penalty box lines) provide reliable high-contrast features in the overlap zone, so calibration should succeed consistently.

### Batch Stitching (~15-20 minutes, unattended)
For each frame pair, apply the saved homography to warp the right image, then blend it with the left image in the overlap zone. Write the stitched frames to the output video.

**Key functions:**
- `cv2.SIFT_create()` or `cv2.ORB_create()` for feature detection
- `cv2.findHomography()` with RANSAC for robust alignment
- `cv2.warpPerspective()` to transform right image into left image's coordinate space
- Linear or multi-band blending in the overlap region for seamless seam

**Calibration file:** `calibrations/YYYY-MM-DD_cal.json` containing:
- Homography matrix (3×3)
- Output canvas dimensions
- Blend region coordinates and weights
- Timecode offset between cameras

**Performance:** The warp + blend per frame is lightweight (~5-10ms per frame on CPU). Decoding two 4K streams is the bottleneck. On an HP Spectre with hardware decode, expect roughly **15-20 minutes** for a 90-minute match. This step is fully unattended.

**Output:** Single stitched panoramic video file, e.g. `match_stitched.mp4` at ~8000×2988 (or whatever the stitched resolution ends up being).

---

## Stage 2: Interactive PTZ Session

**Purpose:** Watch the stitched match video and control a virtual camera using a Nintendo Switch Pro Controller. The app records all inputs to a log file — it does NOT encode video. This keeps the computational load light enough for the Spectre.

### What the app does:
1. **Decodes** the stitched panoramic video and displays it in a window
2. **Reads joystick input** continuously:
   - Left stick X-axis → pan (move crop window left/right)
   - Left stick Y-axis → tilt (move crop window up/down)  
   - Right stick Y-axis or triggers → zoom in/out
3. **Draws a 1920×1080 crop rectangle** on the display to show what the final output frame will look like
4. **Renders a live preview** of the cropped + scoreboard view so you can see exactly what the broadcast will look like
5. **Listens for keyboard events** for scoreboard control
6. **Logs everything** to a timestamped file

### Joystick Mapping (Nintendo Switch Pro Controller):
```
Left Stick X        → Pan left/right
Left Stick Y        → Tilt up/down
ZR (Right Trigger)  → Zoom in (continuous while held)
ZL (Left Trigger)   → Zoom out (continuous while held)
B Button (bottom)   → Snap to center (reset pan)
A Button (right)    → Snap to left goal
Y Button (left)     → Snap to right goal
X Button (top)      → Toggle wide view (full panorama, letterboxed)
```

Note: pygame sees the Switch Pro Controller as a standard gamepad. Button indices
may vary by OS and connection method (Bluetooth vs USB). The config.yaml includes
a button mapping section so you can remap without editing code. Use pygame's
joystick debug output on first run to verify your specific indices.

### Keyboard Mapping (Scoreboard — matches existing OBS control panel shortcuts):
```
Space            → Start/stop match clock (counts up from 0:00)
Q                → Home team goal (+1)
A                → Home team goal (-1, for corrections)
P                → Away team goal (+1)
L                → Away team goal (-1, for corrections)
R                → Reset clock to 0:00
H                → Switch half (1st ↔ 2nd) and reset clock to 0:00
T                → Toggle scoreboard visibility
Escape           → Quit and save log
```

### Smoothing:
- Raw joystick input should be smoothed to prevent jerky camera movement
- Apply an exponential moving average or simple low-pass filter to pan/tilt/zoom values
- The "snap to position" buttons (A, B, X) should animate smoothly over ~0.5 seconds, not jump instantly

### Log File Format:
The log file is a CSV with one row per video frame:

```csv
frame,timestamp,crop_x,crop_y,crop_w,crop_h,home_score,away_score,clock_running,clock_seconds,half,scoreboard_visible
0,0.000,2040,540,1920,1080,0,0,false,0,1,true
1,0.033,2042,540,1920,1080,0,0,true,0,1,true
2,0.067,2045,540,1920,1080,0,0,true,1,1,true
...
6120,204.000,1500,400,2400,1350,1,0,true,204,2,true
```

- `crop_x, crop_y` = top-left corner of the crop window in panorama coordinates
- `crop_w, crop_h` = size of crop window (changes with zoom — always maintains 16:9 aspect ratio)
- At 1.0x zoom: crop is 1920×1080 (native quality). At 1.2x max zoom: crop is 1600×900 (20% upscale)
- At full zoom out (0x): crop is entire panorama (8000×2988), rendered letterboxed in 1920×1080
- Clock counts up from 0:00, resets to 0:00 when half changes
- `half` = 1 or 2 (displayed on scoreboard as "1st" / "2nd")

### Preview Display:
- Show a downscaled version of the full panorama with the crop rectangle overlaid
- Below or beside it, show the actual cropped 1080p view with the scoreboard
- Total window fits on a 1920×1080 laptop screen — perhaps panorama on top (scaled to ~1920×700) and crop preview on bottom (~960×540 scaled)

### Performance Requirements:
- Only decoding one video stream (the stitched panorama) — no encoding
- Display at reasonable frame rate (~24-30fps) — doesn't need to be perfectly smooth, just watchable enough to follow the game and react with the joystick
- Intel Iris Xe hardware decode handles 4K H.264/HEVC fine
- If the stitched video is larger than 4K (8000px wide with 5.3K cameras), we may need to either:
  - (a) Encode the stitch as H.264 4K and accept some quality loss, OR
  - (b) Decode at full resolution but only display downscaled — depends on Iris Xe limits

### Libraries:
- **pygame** for joystick input and display window
- **OpenCV (cv2)** for video decode, crop, resize, and drawing the crop rectangle
- **Pillow (PIL)** for scoreboard rendering (TTF fonts, rounded rectangles, alpha compositing)
- Standard library for CSV logging

---

## Stage 3: Render

**Purpose:** Read the log file from Stage 2 and produce the final 1080p broadcast MP4.

**Method:**
1. Open the stitched panoramic video
2. For each frame, read the corresponding row from the log CSV
3. Crop the panorama at (crop_x, crop_y, crop_w, crop_h)
4. Resize the crop to exactly 1920×1080
5. Composite the scoreboard overlay if scoreboard_visible is true
6. Encode the frame to the output video

**Scoreboard Overlay (scoreboard.py — Pillow renderer):**

Rendered entirely in Python with Pillow. Same renderer used in both the interactive preview (Stage 2) and the final render (Stage 3). No browser, no Playwright, no Chromium dependency.

**Visual Layout (matching existing OBS HTML overlay):**

The scoreboard is a horizontal bar, centered near the bottom of the 1920×1080 frame. Layout left to right:

```
┌──────────────────────────────────────────────────────────────────────┐
│ [H]  HOME     0  │  00:00   │  0     AWAY  [A] │
│ logo  name  score│  timer   │ score  name  logo │
│                  │ 1ST HALF │                   │
│◄─ home color bg ─►◄─ dark ─►◄─ away color bg ──►│
└──────────────────────────────────────────────────────────────────────┘
```

**Dimensions:**
- Bar height: 60px
- Bar width: ~600px (dynamic based on team name lengths, min 600px)
- Border radius: 8px (Pillow rounded rectangle)
- Position: centered horizontally, 50px from bottom of 1920×1080 frame

**Sections (left to right):**

1. **Home team logo** — 45×45px rounded square, semi-transparent white background (`rgba(255,255,255,0.15)` over team color), single initial letter centered, white bold 22px
2. **Home team name** — uppercase, white, bold, 18px, letter-spacing ~1.5px, left-padded 10px from logo, right-padded 15px
3. **Home score box** — dark overlay (`rgba(0,0,0,0.4)` over team color), white bold 32px, centered in 60px wide column, 2px dark border on left
4. **Timer section** — dark gradient background (`#444` to `#222`), 2px border on both sides
   - Clock: `00:00` format, gold color `#FFD700`, bold monospace (Courier New), 24px, letter-spacing 2px
   - Half indicator: `1ST HALF` or `2ND HALF`, gray `#AAAAAA`, uppercase, 10px, 2px below clock
5. **Away score box** — mirror of home score box (dark overlay on away color)
6. **Away team name** — mirror of home (padded, uppercase, white bold)
7. **Away team logo** — mirror of home logo

**Colors:**
- Bar background: dark gradient `#2C2C2C` (top) to `#1A1A1A` (bottom), 95% opacity
- Bar border: `rgba(68,68,68,0.8)`, 2px
- Home section: gradient derived from home team color (user-set in Streamlit), ~90% opacity
  - Default: green gradient `rgba(30,94,58,0.9)` to `rgba(22,68,41,0.9)`
  - Derived: take user's hex color, darken by 20% for gradient bottom
- Away section: same treatment with away team color
  - Default: red gradient `rgba(58,30,30,0.9)` to `rgba(41,22,22,0.9)`
- Score box: team color with `rgba(0,0,0,0.4)` overlay (darkened)
- Timer background: `#444` to `#222` gradient
- Timer text: gold `#FFD700`
- Half indicator text: gray `#AAA`
- All other text: white `#FFFFFF`
- Text shadows: 2px offset, black, ~80% opacity (subtle depth)

**Fonts:**
- Team names, scores, logos: Arial Bold (or Helvetica Bold fallback)
- Timer digits: Courier New Bold (monospace for stable width as digits change)
- Half indicator: Arial, 10px
- Bundle `arial.ttf` and `cour.ttf` in `assets/fonts/` for consistent rendering across systems

**Rendering approach:**
1. Create a transparent RGBA image (1920×1080)
2. Draw the bar background as a rounded rectangle
3. Draw each section left-to-right with appropriate background fills
4. Draw text with Pillow `ImageDraw.text()` using loaded TTF fonts
5. Apply text shadow by drawing text offset in black first, then white on top
6. Cache the rendered image — only re-render when state changes (score, clock second, half, visibility)
7. Composite cached scoreboard onto video frame with alpha blending

**Performance:**
- Pillow render: ~1-2ms per scoreboard frame
- Cache invalidation: re-render on any state change (score, clock second, half, visibility toggle)
- During interactive session: same renderer, no performance concern
- During final render: scoreboard re-renders at most once per second (when clock ticks), cached for all 30 frames in between

**Interactive session (Stage 2) scoreboard preview:**
- Same Pillow renderer as the final render — no separate preview needed
- Scoreboard is drawn directly onto the pygame display surface each frame
- Convert Pillow Image to pygame Surface via `pygame.image.fromstring()`

**Encoding:**
```
ffmpeg -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 30 -i pipe:0 \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  output_broadcast.mp4
```
Or use OpenCV's VideoWriter. ffmpeg pipe gives more control over quality.

**Audio:**
- Extract audio from the **left camera** source file
- Mux it into the final output, time-aligned with the video
- Both cameras are mounted side by side so audio is effectively identical

**Performance:**
- Decode + crop + overlay + encode, no display
- Single-threaded CPU encode at libx264 preset medium
- Expected time: **30-60 minutes** for a 90-minute match on the Spectre
- Fully unattended — start it and walk away

**Output:** `match_broadcast_1080p.mp4` — the finished product.

---

## Stage 4 (Optional): Replay Clips in Clipchamp

If there's a great goal or save worth replaying:

1. Open the stitched panoramic source video in Clipchamp (or the raw GoPro files)
2. Navigate to the moment
3. Crop to the relevant area (goal mouth, player, etc.)
4. Optionally slow it down (0.5x or 0.25x)
5. Export the short clip
6. Open the finished broadcast MP4 in Clipchamp
7. Split at the insertion point (right after the goal in real-time)
8. Insert the replay clip
9. Export final version

This is standard video editing — Clipchamp handles it fine and it's already installed on Windows 11.

---

## Streamlit User Interface

The system is controlled through a Streamlit app (`app.py`) with a sidebar for match status and three pages for each stage. The interactive PTZ session (Stage 2) launches as a separate pygame window since it requires real-time controller input — Streamlit acts as mission control.

Run with:
```
streamlit run app.py
```

### Sidebar (persistent across all pages)

Always visible. Shows at-a-glance match status:

```
───────────────────────────
  Match: 2026-03-15
  Eagles vs Hawks
───────────────────────────
  ✓ Stitched video
  ✗ PTZ session log
  ✗ Final broadcast
───────────────────────────
```

Status checkmarks update automatically based on which output files exist for the current match date.

### Page 1 — Calibrate + Stitch

**File Selection:**
- Two file upload buttons: "Left Camera" and "Right Camera"
- After selection, auto-reads file metadata via ffprobe and displays:
  - Resolution (e.g. 5312×2988)
  - Duration (e.g. 1:27:14)
  - Frame rate (e.g. 30 fps)
  - Sync method: "Timecode sync detected" or "No timecode — will use audio sync"
  - Timecode/audio offset between cameras

**Time Estimate:**
- Calculated from file duration: roughly `duration × 0.2` for stitch time
- Displayed as: "Estimated time: ~18 minutes"

**Start Button:**
- "Calibrate + Stitch" button kicks off processing
- Progress bar showing current frame / total frames
- Stage indicator: "Syncing..." → "Calibrating..." → "Stitching frame 12,400 / 156,000"
- When complete: success message + file path to stitched video

**Stitch Preview:**
- After stitching completes, display a thumbnail of a single stitched panorama frame
- Lets you visually verify the stitch looks correct (seam alignment, overlap blend) before moving to Stage 2
- If the stitch looks off, you can re-run without wasting time on the interactive session

### Page 2 — Interactive PTZ Session

**Match Setup (inputs saved to match config):**
- Home team name (text input, default from config.yaml)
- Away team name (text input)
- Home team color (color picker)
- Away team color (color picker)

These values are passed to the pygame session on launch and also used by the render stage for the final scoreboard overlay.

**Controls Reference Card:**

Displayed as a two-column reference so it's visible before and during the session:

```
┌─────────────────────────────────────────────────────────┐
│  CAMERA (Switch Pro Controller)  │  SCOREBOARD (Keyboard)│
│─────────────────────────────────│───────────────────────│
│  Left Stick      Pan / Tilt     │  Space    Start/Stop  │
│  ZR trigger      Zoom in        │  Q / A    Home +1 / -1│
│  ZL trigger      Zoom out       │  P / L    Away +1 / -1│
│  B button        Snap center    │  H        Switch half  │
│  A button        Snap left goal │            + reset clock│
│  Y button        Snap right goal│  R        Reset clock  │
│  X button        Wide view      │  T        Toggle board │
│                                 │  Esc      Quit + save  │
└─────────────────────────────────────────────────────────┘
```

**Launch Button:**
- "Launch Interactive Session" button spawns pygame as a subprocess
- Streamlit page shows: "Session in progress... close the pygame window when done"
- When pygame exits, page refreshes and displays session summary:
  - Session duration
  - Final score
  - Number of score changes
  - Log file path

### Page 3 — Render

**Pre-Render Summary:**
- Match duration (from log)
- Final score (from log)
- Scoreboard settings (team names, colors — from Page 2 inputs)
- Estimated render time (roughly `duration × 0.5`)

**Render Button:**
- "Render Final Broadcast" button starts encoding
- Progress bar: "Rendering frame 12,400 / 156,000"
- When complete: success message + file path
- "Open output folder" convenience button

---

## Project File Structure

```
soccer-broadcast/
├── README.md
├── requirements.txt          # opencv-python, pygame, numpy, Pillow, streamlit
├── config.yaml               # default team names, colors, controller mappings
│
├── app.py                    # Streamlit entry point (sidebar + page routing)
├── pages/
│   ├── 1_Stitch.py           # Page 1: file selection, calibrate + stitch
│   ├── 2_Interactive.py      # Page 2: match setup, controls reference, launch pygame
│   └── 3_Render.py           # Page 3: render summary, progress, output
│
├── calibrate.py              # Per-game: compute stitch homography from frame pair
├── sync_audio.py             # Fallback: audio cross-correlation if timecode sync missing
├── stitch.py                 # Stage 1: calibrate + sync + batch stitch two videos into panorama
├── interactive.py            # Stage 2: pygame joystick + preview + log recording
├── render.py                 # Stage 3: read log, crop, overlay, encode final MP4
│
├── scoreboard.py             # Scoreboard rendering module (renders HTML overlay to PNG)
├── smoother.py               # Joystick input smoothing utilities
│
├── calibrations/             # Per-game stitch calibration files
│   └── 2026-03-01_cal.json
│
├── logs/                     # Match log files from interactive sessions
│   └── 2026-03-01_match.csv
│
├── output/                   # Final broadcast MP4s
│   └── 2026-03-01_broadcast_1080p.mp4
│
└── assets/
    ├── overlay/
    │   ├── scoreboard-display.html   # Original OBS overlay (kept for reference only)
    │   └── scoreboard-control.html   # OBS control panel (kept for reference only)
    └── fonts/
        ├── arial.ttf                 # Team names, scores, logos
        └── cour.ttf                  # Timer digits (monospace)
```

---

## Config File (config.yaml)

```yaml
# Team settings (defaults — can be overridden per-match in Streamlit)
home_team: "Eagles"
away_team: "Hawks"
home_color: "#1E3A5F"         # dark blue (hex)
away_color: "#8B0000"         # dark red (hex)

# Video settings
output_width: 1920
output_height: 1080
output_fps: 30
output_crf: 18

# Joystick settings (Nintendo Switch Pro Controller)
# Button/axis indices may vary — run interactive.py with --debug-controller to check yours
controller:
  axis_pan: 0              # Left stick X
  axis_tilt: 1             # Left stick Y
  axis_zoom_in: 5          # ZR (right trigger)
  axis_zoom_out: 4         # ZL (left trigger)
  button_snap_center: 1    # B button (bottom)
  button_snap_left: 0      # A button (right)
  button_snap_right: 3     # Y button (left)
  button_wide_view: 2      # X button (top)
  deadzone: 0.10           # ignore stick input below this threshold

pan_speed: 5.0            # pixels per frame at full stick deflection
tilt_speed: 3.0
zoom_speed: 0.02          # zoom factor change per frame at full trigger
min_zoom: 0              # 0 = full panorama (letterboxed, 2.68:1 aspect ratio)
max_zoom: 1.2             # 20% upscale max — crop = 1600×900, no visible quality loss
smoothing_factor: 0.15    # exponential smoothing (0 = no smoothing, 1 = frozen)

# Snap positions (in panorama pixel coordinates)
snap_center_x: 4000       # center of ~8000px panorama (adjust after seeing stitched output)
snap_left_goal_x: 500
snap_right_goal_x: 7500
snap_y: 1494              # vertical center of 2988px panorama

# Scoreboard
scoreboard_position: "bottom"  # "top" or "bottom"
scoreboard_offset: 50          # pixels from top/bottom edge
scoreboard_font: "assets/fonts/arial.ttf"
scoreboard_mono_font: "assets/fonts/cour.ttf"
```

---

## Build Order

Build and test in this sequence:

### Phase 1: Stitch Pipeline
1. **calibrate.py** — Extract frames, detect features, compute homography, save per-game calibration
2. **stitch.py** — Read calibration + timecode offset, batch stitch full match video
3. Test with short sample clips from both GoPros before running on full match

### Phase 2: Interactive Viewer
4. **interactive.py (video only)** — Load stitched video, display with crop rectangle, no joystick yet
5. **Add joystick** — Read Switch Pro Controller, move crop window, display live preview
6. **smoother.py** — Add input smoothing so camera movement feels natural
7. **Add scoreboard** — Keyboard events update score/clock, render existing HTML overlay on preview
8. **Add logging** — Write CSV log file every frame
9. Test full interactive session on a sample match

### Phase 3: Render Pipeline
10. **render.py** — Read log CSV, decode source, crop, overlay scoreboard, encode output
11. **Add audio** — Mux audio track from source into final output
12. Test end-to-end: record interactive session → render → watch output

### Phase 4: Streamlit UI
13. **app.py** — Sidebar with match status, page routing
14. **pages/1_Stitch.py** — File pickers, metadata display, time estimate, progress bar
15. **pages/2_Interactive.py** — Match setup inputs (team names, colors), controls reference card, launch pygame as subprocess
16. **pages/3_Render.py** — Pre-render summary, progress bar, output download
17. Test full workflow through Streamlit: stitch → interactive → render

### Phase 5: Polish
18. Config file support (config.yaml)
19. Error handling and edge cases (joystick disconnects, end of video, etc.)
20. Snap-to-position animations
21. README with full setup and usage instructions

---

## Dependencies

```
Python 3.10+
opencv-python (or opencv-python-headless for render-only)
streamlit
pygame
numpy
scipy               # for audio cross-correlation fallback sync
Pillow
PyYAML
```

Optional:
```
ffmpeg (system install) — for audio extraction, final muxing, and potentially faster encode
```

---

## Key Design Decisions

1. **Two-pass approach (interactive + render):** The HP Spectre with Intel Iris Xe cannot decode a large video while simultaneously encoding 1080p output. Splitting into "log the inputs" and "render from the log" keeps each pass within the laptop's capability.

2. **Flat panorama, not equirectangular:** Since we're stitching two standard rectilinear GoPro videos (not 360° cameras), the output is a normal flat wide video. No special projection math or v360 filters needed. Just a homography warp and blend.

3. **Per-game calibration:** Even with a rigid dual-camera bracket, small variations in tripod placement and camera tightening between games can shift the stitch geometry. Calibration runs automatically at the start of each game's processing (~10-15 seconds) and saves to a per-game JSON file. No manual intervention needed.

4. **GoPro timecode sync with audio fallback:** The Hero 13 supports cross-device timecode sync via the Quik app — frame alignment is read directly from file metadata via ffprobe. If timecode sync was forgotten at the field, the system auto-detects this and falls back to audio cross-correlation using ambient sound.

5. **CSV log format:** Simple, human-readable, easy to debug. Can be manually edited if you want to tweak a camera move without re-watching the whole match. Can also be visualized or analyzed after the fact.

6. **Pillow scoreboard renderer:** The scoreboard is rendered natively in Python using Pillow, matching the visual design of the existing OBS HTML overlay (dark bar, team-colored sections, gold timer, half indicator). One renderer used everywhere — interactive preview and final render are identical. No browser dependency. The same keyboard shortcuts from the OBS control panel (Space, Q/A, P/L, R, H) are used in the interactive session so muscle memory transfers.

7. **Replays in Clipchamp, not in the Python app:** Adding replay insertion into the Python app would significantly increase complexity (timeline management, secondary video sources, transitions). Clipchamp already does this well and is free on Windows 11. Keep the Python app focused on the main broadcast pass.
