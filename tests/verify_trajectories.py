import cv2
import numpy as np
import subprocess
import os

trajectories = ['dolly-zoom-in', 'zoom-in', 'circle', 'swing']

print("\n" + "=" * 65)
print("📊 [Comprehensive Camera Motion Analysis: All 4 Trajectories]")
print("=" * 65)

for traj in trajectories:
    orig_name = f"orig_{traj}.mp4"
    curr_path = f"outputs/videos/moon_{traj}.mp4"
    
    # 1. 本家動画の抽出
    if not os.path.exists(orig_name):
        try:
            orig_bytes = subprocess.check_output(["git", "show", f"de04467:video/moon_{traj}.mp4"])
            with open(orig_name, "wb") as f:
                f.write(orig_bytes)
        except Exception as e:
            print(f"[{traj}] Skip (Original video not found in git): {e}")
            continue

    def track(path):
        cap = cv2.VideoCapture(path)
        xs, ys, areas = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (960, 960))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                xs.append(M["m10"] / M["m00"])
                ys.append(M["m01"] / M["m00"])
                areas.append(M["m00"])
            else:
                xs.append(0); ys.append(0); areas.append(0)
        cap.release()
        return np.array(xs), np.array(ys), np.array(areas)

    orig_xs, orig_ys, orig_areas = track(orig_name)
    curr_xs, curr_ys, curr_areas = track(curr_path)

    n = min(len(orig_xs), len(curr_xs))
    dx_o = np.max(np.abs(orig_xs[:n] - orig_xs[0]))
    dx_c = np.max(np.abs(curr_xs[:n] - curr_xs[0]))
    
    dy_o = np.max(np.abs(orig_ys[:n] - orig_ys[0]))
    dy_c = np.max(np.abs(curr_ys[:n] - curr_ys[0]))

    zoom_o = np.max(np.abs((orig_areas[:n] - orig_areas[0]) / orig_areas[0])) * 100.0
    zoom_c = np.max(np.abs((curr_areas[:n] - curr_areas[0]) / curr_areas[0])) * 100.0

    print(f"\n🎥 Trajectory: [{traj}] (Total Frames: {n})")
    print(f"   X-Motion (Left/Right) : Orig = {dx_o:5.2f} px | Curr = {dx_c:5.2f} px  (Ratio: {dx_c/dx_o*100.0:5.1f}%)" if dx_o > 0 else "   X-Motion: 0 px")
    print(f"   Y-Motion (Up/Down)    : Orig = {dy_o:5.2f} px | Curr = {dy_c:5.2f} px  (Ratio: {dy_c/dy_o*100.0:5.1f}%)" if dy_o > 0 else "   Y-Motion: 0 px")
    print(f"   Forward Zoom Scale    : Orig = {zoom_o:5.2f} %  | Curr = {zoom_c:5.2f} %   (Ratio: {zoom_c/zoom_o*100.0:5.1f}%)" if zoom_o > 0 else "   Zoom: 0 %")

    if os.path.exists(orig_name):
        os.remove(orig_name)

print("\n" + "=" * 65 + "\n")