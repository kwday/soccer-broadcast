# Soccer Broadcast System

A DIY broadcast pipeline that stitches two GoPro camera feeds into a panoramic video, provides interactive PTZ (pan/tilt/zoom) control with a game controller, and renders a final broadcast-quality output with scoreboard overlay and audio.

## Hardware

- **Cameras:** 2x GoPro Hero 13 at 5.3K (5312x2988), standard wide lens (~155 FOV)
- **Mount:** Dual GoPro bracket on tall tripod (10-13 ft) at midfield
- **Controller:** Nintendo Switch Pro Controller (USB or Bluetooth)
- **Overlap:** 20-30% overlap between cameras for clean stitch seam

## Pipeline

### Stage 1: Calibrate + Stitch (automated, ~15-20 min)

Syncs two camera feeds (timecode or audio cross-correlation), calibrates the stitch (SIFT features + RANSAC homography), and blends them into a single ~8000x2988 panoramic video.

### Stage 2: Interactive PTZ Session (interactive, ~match length)

Watch the panorama and control a virtual camera with the Switch Pro Controller. Left stick pans/tilts, triggers zoom, face buttons snap to preset positions (center, goals, wide). Keyboard controls the scoreboard (score, clock, half). All inputs are logged to CSV.

### Stage 3: Render (automated, ~30-60 min)

Reads the session log, crops each frame from the panorama, composites the scoreboard overlay, encodes to 1920x1080 H.264, and muxes audio from the left camera.

## Setup

### Requirements

- Python 3.10+
- FFmpeg (installed automatically via `imageio-ffmpeg`)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Edit `config.yaml` to set:
- Team names and colors
- Controller button/axis mappings (run `python interactive.py --video test.mp4 --debug-controller` to check yours)
- Pan/tilt/zoom speeds
- Snap-to-position coordinates

### Run

```bash
streamlit run app.py
```

Or use the command-line tools directly:

```bash
# Stage 1: Stitch
python stitch.py --left left.mp4 --right right.mp4 --output output/stitched.mp4

# Stage 2: Interactive
python interactive.py --video output/stitched.mp4 --log-output logs/match.csv

# Stage 3: Render
python render.py --video output/stitched.mp4 --log logs/match.csv --output output/broadcast.mp4 --audio left.mp4
```

## Controls

| Camera (Switch Pro Controller) | Scoreboard (Keyboard) |
|:---|:---|
| Left Stick - Pan / Tilt | Space - Start/Stop clock |
| ZR trigger - Zoom in | Q / A - Home +1 / -1 |
| ZL trigger - Zoom out | P / L - Away +1 / -1 |
| B button - Snap center | H - Switch half + reset clock |
| A button - Snap left goal | R - Reset clock |
| Y button - Snap right goal | T - Toggle scoreboard |
| X button - Wide view | Esc - Quit + save |

## Project Structure

```
soccer-broadcast/
├── app.py                    # Streamlit entry point
├── config.yaml               # Default settings
├── calibrate.py              # Per-game stitch calibration
├── sync_audio.py             # Audio cross-correlation sync
├── stitch.py                 # Batch panorama stitching
├── interactive.py            # Interactive PTZ viewer
├── smoother.py               # Input smoothing
├── scoreboard.py             # Pillow scoreboard renderer
├── render.py                 # Final broadcast encoder
├── pages/
│   ├── 1_Stitch.py           # Streamlit stitch page
│   ├── 2_Interactive.py      # Streamlit interactive page
│   └── 3_Render.py           # Streamlit render page
├── calibrations/             # Per-game calibration files
├── logs/                     # Session log files
├── output/                   # Final broadcast videos
└── assets/fonts/             # Scoreboard fonts
```

## Recording Checklist

1. Mount both GoPros in dual bracket on tripod at midfield
2. Sync timecode via GoPro Quik app
3. Lock all settings to manual (resolution, frame rate, white balance, exposure)
4. Start both cameras, run for full match (~85 min with buffer)
5. Pull SD cards after recording

## Testing

```bash
python -m pytest tests/ -v
```
