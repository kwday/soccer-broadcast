# Build Checklist

> **Instructions for Claude:** Read this file at the start of every session.
> 
> **Workflow for each step:**
> 1. Build the code
> 2. Write and run tests to verify it works
> 3. Check off the build task AND the test task
> 4. `git add . && git commit -m "step N: description" && git push`
> 5. Check off the commit task
> 6. Update "Current Status" at the bottom
> 7. Move to next step
>
> **Do not move to the next step until the current step's tests pass.**
> Refer to DESIGN.md for full specifications.

---

## Phase 1: Stitch Pipeline

- [x] 1. **calibrate.py** — Extract frames from left/right video, SIFT/ORB feature detection on overlap region, compute homography with RANSAC, save to `calibrations/YYYY-MM-DD_cal.json`
- [x] 1t. **Test** — Run on two sample images with known overlap, verify homography output, verify JSON file is written with correct structure
- [x] 1c. **Commit** — `git commit -m "step 1: calibrate.py - per-game stitch calibration"`

- [x] 2. **sync_audio.py** — Fallback audio cross-correlation (extract audio via ffmpeg, scipy cross-correlate, output offset in seconds)
- [x] 2t. **Test** — Run on two audio files with known offset, verify detected offset matches expected within 1 frame
- [x] 2c. **Commit** — `git commit -m "step 2: sync_audio.py - fallback audio sync"`

- [x] 3. **stitch.py** — Auto-detect timecode vs audio sync, load per-game calibration, batch warp+blend all frames, output stitched panorama video
- [x] 3t. **Test** — Run on short clip pair (~10 seconds), verify output video exists, verify resolution matches expected panorama dimensions, spot-check stitch seam visually
- [x] 3c. **Commit** — `git commit -m "step 3: stitch.py - batch panorama stitching"`

- [x] 4. **Integration test** — Run full Phase 1 pipeline on sample clips (calibrate → sync → stitch), verify end-to-end output
- [x] 4c. **Commit** — `git commit -m "step 4: phase 1 integration test passed"`

## Phase 2: Interactive Viewer

- [x] 5. **interactive.py (video only)** — Load stitched video with OpenCV, display in pygame window with crop rectangle overlay
- [x] 5t. **Test** — Opens window, displays video frames, crop rectangle visible, arrow keys or mouse move crop, Escape quits cleanly
- [x] 5c. **Commit** — `git commit -m "step 5: interactive.py - video display with crop rect"`

- [x] 6. **Add joystick** — Read Switch Pro Controller via pygame, move crop window with left stick, zoom with triggers
- [x] 6t. **Test** — Print joystick axis/button values to console, verify crop moves with stick, zoom works with triggers, deadzone filters noise
- [x] 6c. **Commit** — `git commit -m "step 6: joystick input for pan/tilt/zoom"`

- [x] 7. **smoother.py** — Exponential moving average on joystick input, smooth snap-to-position animations (~0.5s)
- [x] 7t. **Test** — Unit test: feed step input, verify output ramps smoothly. Visual test: snap-to-center animates, no jerky movement
- [x] 7c. **Commit** — `git commit -m "step 7: smoother.py - input smoothing"`

- [x] 8. **scoreboard.py** — Full Pillow renderer matching OBS overlay design (see DESIGN.md scoreboard spec: 60px bar, team colors, gold timer, half indicator, fonts)
- [x] 8t. **Test** — Render scoreboard to PNG with sample state, open and visually compare to OBS HTML overlay. Test all states: 0-0, 3-2, half change, visibility toggle
- [x] 8c. **Commit** — `git commit -m "step 8: scoreboard.py - Pillow scoreboard renderer"`

- [x] 9. **Add scoreboard to interactive** — Keyboard events (Space, Q/A, P/L, H, R, T) update score/clock/half state, composite Pillow scoreboard onto pygame preview
- [x] 9t. **Test** — Launch interactive, press each key, verify score increments/decrements, clock starts/stops/resets, half switches and resets clock, scoreboard toggles visibility
- [x] 9c. **Commit** — `git commit -m "step 9: scoreboard keyboard controls in interactive"`

- [x] 10. **Add logging** — Write CSV log every frame (frame, timestamp, crop coords, score, clock, half, visibility)
- [x] 10t. **Test** — Run short interactive session, verify CSV file written, verify column count, verify timestamps increment, verify score changes appear at correct frames
- [x] 10c. **Commit** — `git commit -m "step 10: CSV frame logging"`

- [ ] 11. **Integration test** — Run full interactive session on sample stitched video, verify all controls work together, verify log file is complete
- [ ] 11c. **Commit** — `git commit -m "step 11: phase 2 integration test passed"`

## Phase 3: Render Pipeline

- [ ] 12. **render.py** — Read log CSV, decode stitched video, crop per log, composite scoreboard, encode 1080p output
- [ ] 12t. **Test** — Render from a known log CSV, verify output is 1920x1080, verify frame count matches log rows, verify scoreboard visible in output frames, spot-check crop positions
- [ ] 12c. **Commit** — `git commit -m "step 12: render.py - final broadcast encoder"`

- [ ] 13. **Add audio** — Extract audio from left camera source, mux into final output via ffmpeg
- [ ] 13t. **Test** — Verify final MP4 has audio track, verify audio duration matches video duration, play back and verify sync
- [ ] 13c. **Commit** — `git commit -m "step 13: audio mux into final output"`

- [ ] 14. **End-to-end test** — Full pipeline: stitch → interactive → render → watch final output
- [ ] 14c. **Commit** — `git commit -m "step 14: phase 3 end-to-end test passed"`

## Phase 4: Streamlit UI

- [ ] 15. **app.py** — Streamlit entry point, sidebar with match status (date, teams, ✓/✗ per stage)
- [ ] 15t. **Test** — `streamlit run app.py` launches, sidebar displays, status checkmarks reflect file existence
- [ ] 15c. **Commit** — `git commit -m "step 15: app.py - streamlit shell with sidebar"`

- [ ] 16. **pages/1_Stitch.py** — File pickers, metadata display, timecode vs audio sync detection, time estimate, progress bar, stitch preview thumbnail after completion
- [ ] 16t. **Test** — Select files, verify metadata displays correctly, run stitch, verify progress bar updates, verify thumbnail appears
- [ ] 16c. **Commit** — `git commit -m "step 16: stitch page with preview thumbnail"`

- [ ] 17. **pages/2_Interactive.py** — Match setup (team names, colors), controls reference card, launch pygame as subprocess, session summary on return
- [ ] 17t. **Test** — Set team names/colors, launch interactive, verify settings pass through, verify summary appears after pygame closes
- [ ] 17c. **Commit** — `git commit -m "step 17: interactive setup page"`

- [ ] 18. **pages/3_Render.py** — Pre-render summary (score, duration), time estimate, progress bar, output file path
- [ ] 18t. **Test** — Verify summary reads from log correctly, run render, verify progress bar, verify output path shown
- [ ] 18c. **Commit** — `git commit -m "step 18: render page"`

- [ ] 19. **Streamlit integration test** — Full workflow through Streamlit: stitch → interactive → render
- [ ] 19c. **Commit** — `git commit -m "step 19: phase 4 streamlit integration test passed"`

## Phase 5: Polish

- [ ] 20. **config.yaml** — Verify all defaults work, document each setting with comments
- [ ] 20c. **Commit** — `git commit -m "step 20: config.yaml finalized"`

- [ ] 21. **Error handling** — Joystick disconnect recovery, end-of-video handling, missing files, failed calibration graceful fallback
- [ ] 21t. **Test** — Simulate each error condition, verify app doesn't crash, verify user sees helpful message
- [ ] 21c. **Commit** — `git commit -m "step 21: error handling"`

- [ ] 22. **requirements.txt** — Pin versions, verify clean install in fresh venv
- [ ] 22t. **Test** — `pip install -r requirements.txt` in clean venv, run `streamlit run app.py`, verify launches
- [ ] 22c. **Commit** — `git commit -m "step 22: requirements.txt pinned"`

- [ ] 23. **README.md** — Setup instructions, hardware list, usage walkthrough
- [ ] 23c. **Commit** — `git commit -m "step 23: README.md - project complete"`

---

## Current Status
**Last completed:** Step 10 — CSV frame logging
**Currently working on:** Step 11 — Phase 2 integration test
**Blockers/Notes:** (none)
