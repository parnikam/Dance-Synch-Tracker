import streamlit as st
import cv2, numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.pose_extraction import iter_video_frames, extract_keypoints_bgr_frame, mp_pose
from src.kinematics import compute_joint_angles, velocities, center_of_mass, jitter_bands, L
from src.visuals import draw_skeleton, Trail, overlay_bands_timeline
from src.utils import concat_side_by_side, resize_keep_ar

st.set_page_config(page_title="Pose Explorer", layout="wide")

st.title("Pose Explorer — No Scores, Just Insights")

uploaded = st.file_uploader("Upload a dance video (mp4/mov)", type=["mp4", "mov", "m4v", "avi"])
col_preview, col_timelines = st.columns([2,1])

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
        tmp.write(uploaded.read())
        vid_path = tmp.name

    # Controls
    max_frames = st.slider("Process first N frames (for speed)", 120, 3000, 900, step=60)
    trail_len = st.slider("Trail length (frames)", 10, 120, 40, step=5)

    # Storage
    frames = []
    angles_series = {k: [] for k in ["L_ELBOW","R_ELBOW","L_KNEE","R_KNEE","L_HIP","R_HIP","L_SHOULDER","R_SHOULDER"]}
    com_series = []

    wrist_trail_L = Trail(maxlen=trail_len)
    wrist_trail_R = Trail(maxlen=trail_len)
    foot_trail_L = Trail(maxlen=trail_len)
    foot_trail_R = Trail(maxlen=trail_len)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose_model:
        for frame, idx, fps in iter_video_frames(vid_path, max_frames=max_frames):
            res = extract_keypoints_bgr_frame(frame, pose_model)
            if res is None:
                frames.append(frame)
                continue
            xy, vis, xyz = res

            # Trails
            wrist_trail_L.push(xy[L['L_WRIST']]); wrist_trail_R.push(xy[L['R_WRIST']])
            foot_trail_L.push(xy[L['L_ANKLE']]); foot_trail_R.push(xy[L['R_ANKLE']])

            # Angles and CoM
            ang = compute_joint_angles(xy)
            for k, v in ang.items():
                angles_series[k].append(v)
            com = center_of_mass(xy)
            com_series.append(com)

            # Overlays
            skel = draw_skeleton(frame, xy, color=(255,255,255))
            skel = wrist_trail_L.draw(skel, (180,180,255))
            skel = wrist_trail_R.draw(skel, (180,255,255))
            skel = foot_trail_L.draw(skel, (180,255,180))
            skel = foot_trail_R.draw(skel, (255,180,180))

            # Balance line (vertical line through CoM)
            h, w = skel.shape[:2]
            cv2.circle(skel, tuple(np.int32(com)), 6, (0,255,0), -1, cv2.LINE_AA)
            cv2.line(skel, (int(com[0]), 0), (int(com[0]), h), (0,120,0), 1, cv2.LINE_AA)

            frames.append(skel)

    if len(frames) > 0:
        out_path = "overlayed_video.mp4"   # save next to your script
        fps_val = fps if 'fps' in locals() else 30.0
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4
        writer = cv2.VideoWriter(out_path, fourcc, fps_val, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"✅ Saved: {out_path}")

    # Build timelines + bands (green/amber/red)
    fps = fps if 'fps' in locals() else 30.0
    import math
    timelines = []
    for name, series in angles_series.items():
        if len(series) < 5:
            continue
        arr = np.array(series, dtype=np.float32)
        # derive band labels using jitter on angle accel
        bands = jitter_bands(arr, fps=fps, w_seconds=0.4, a_thresh=(120, 300))
        timelines.append((name, arr, bands))

    with col_preview:
        # Side-by-side: original vs overlay preview (we only have overlay frames here)
        st.subheader("Playback with Pose Overlay, Trails, and Balance Line")
        # Create a small preview GIF-like by sampling frames
        step = max(1, len(frames)//200)
        preview = [resize_keep_ar(f, width=640) for f in frames[::step]]
        # Streamlit can't play np arrays as video easily; show as image sequence
        st.image(preview, use_column_width=True)

    with col_timelines:
        st.subheader("Timelines with Green/Amber/Red Bands")
        width = 420
        height = 40
        for name, arr, bands in timelines:
            bar = overlay_bands_timeline(width, height, bands)
            st.markdown(f"**{name}**")
            st.image(bar, use_column_width=False)
        st.caption("Bands reflect local acceleration magnitude, highlighting jittery or unstable motion segments.")

    st.success("Done. Tip: lower resolution or shorten N frames if performance is slow.")
else:
    st.info("Upload a short clip to begin. 10–20 seconds at 540p works well for a quick preview.")
