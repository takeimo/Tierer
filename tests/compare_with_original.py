import os
import cv2
import numpy as np
import subprocess

print("[Compare] Extracting original video from git commit de04467 ...")
try:
    orig_bytes = subprocess.check_output(["git", "show", "de04467:video/moon_swing.mp4"])
    with open("orig_moon_swing.mp4", "wb") as f:
        f.write(orig_bytes)
except Exception as e:
    print(f"Failed to extract original video: {e}")
    exit(1)

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from {video_path}")
    return frame

orig_frame = get_first_frame("orig_moon_swing.mp4")
current_frame = get_first_frame("outputs/videos/moon_swing.mp4")

h_orig, w_orig = orig_frame.shape[:2]
h_curr, w_curr = current_frame.shape[:2]

print("\n" + "=" * 55)
print("📊 [Frame Resolution Comparison]")
print("=" * 55)
print(f"Original Video Resolution : {w_orig} x {h_orig}")
print(f"Current Video Resolution  : {w_curr} x {h_curr}")

# サイズが異なる場合はリサイズして比較
if (w_orig, h_orig) != (w_curr, h_curr):
    print(f"[Notice] Resolutions differ! Scaling ratio: W={w_curr/w_orig:.3f}, H={h_curr/h_orig:.3f}")
    curr_resized = cv2.resize(current_frame, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
else:
    curr_resized = current_frame

diff = cv2.absdiff(orig_frame, curr_resized)
mean_diff = np.mean(diff)

orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

shift, response = cv2.phaseCorrelate(orig_gray, curr_gray)
dx, dy = shift

print(f"Mean Pixel Difference     : {mean_diff:.2f} / 255.0")
print(f"Detected X-Shift (dx)     : {dx:+.2f} pixels (正: 右ズレ / 負: 左ズレ)")
print(f"Detected Y-Shift (dy)     : {dy:+.2f} pixels (正: 下ズレ / 負: 上ズレ)")
print("=" * 55 + "\n")

cv2.imwrite("orig_frame.png", orig_frame)
cv2.imwrite("current_frame.png", current_frame)
cv2.imwrite("comparison_diff.png", diff)
print("Saved: orig_frame.png, current_frame.png, comparison_diff.png")

if os.path.exists("orig_moon_swing.mp4"):
    os.remove("orig_moon_swing.mp4")