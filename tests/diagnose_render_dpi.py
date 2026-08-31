import os
import cv2
import numpy as np
import yaml
import torch
import copy
import imageio.v2 as imageio
from mesh import read_ply, Canvas_view

# 設定とメッシュの読み込み
config = yaml.safe_load(open("argument.yml", "r"))
mesh_fi = "outputs/3d_meshes/moon.ply"
if not os.path.exists(mesh_fi):
    print("Error: outputs/3d_meshes/moon.ply not found. Please run Tierer first.")
    exit(1)

verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)

fov = max(hFov, vFov) * 180. / np.pi
canvas_size = max(Height, Width)

# 1. VisPy キャンバスの初期化と生レンダリング (Stage 0)
canvas = Canvas_view(fov, verts, faces, colors, canvas_size=canvas_size)
raw_render = canvas.render()
cv2.imwrite("stage0_raw_render.png", cv2.cvtColor(raw_render[..., :3], cv2.COLOR_RGB2BGR))

# 2. anchor クロップ (Stage 1)
aspect_ratio = float(Height) / float(Width)
if aspect_ratio <= 1:
    img_w_len = Width
    img_h_len = img_w_len * aspect_ratio
    anchor = [int(max(0, int((canvas_size)//2 - img_h_len//2))),
              int(min(int((canvas_size)//2 + img_h_len//2), canvas_size-1)),
              0,
              canvas_size]
stage1_anchor = raw_render[anchor[0]:anchor[1], anchor[2]:anchor[3], :3]
cv2.imwrite("stage1_anchor.png", cv2.cvtColor(stage1_anchor, cv2.COLOR_RGB2BGR))

# 3. border クロップ (Stage 2)
# sample['int_mtx'] 相当の計算
int_mtx = np.array([[max(Height, Width), 0, Width//2], [0, max(Height, Width), Height//2], [0, 0, 1]], dtype=np.float32)
int_mtx[0, :] /= float(Width)
int_mtx[1, :] /= float(Height)

top = int(Height // 2 - int_mtx[1, 2] * Height)
left = int(Width // 2 - int_mtx[0, 2] * Width)
down = int(int_mtx[1, 2] * Height + Height // 2)
right = int(int_mtx[0, 2] * Width + Width // 2)
border = [top, down, left, right]

stage2_border = stage1_anchor[border[0]:border[1], border[2]:border[3]]
cv2.imwrite("stage2_border.png", cv2.cvtColor(stage2_border, cv2.COLOR_RGB2BGR))

# 4. crop_border クロップ & リサイズ (Stage 3)
H_c, W_c, _ = stage2_border.shape
o_t = int(H_c * config['crop_border'][0])
o_l = int(W_c * config['crop_border'][1])
o_b = int(H_c * config['crop_border'][2])
o_r = int(W_c * config['crop_border'][3])
stage3_cropped = stage2_border[o_t:H_c-o_b, o_l:W_c-o_r]
stage3_final = cv2.resize(stage3_cropped, (W_c, H_c), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("stage3_final.png", cv2.cvtColor(stage3_final, cv2.COLOR_RGB2BGR))

print("\n" + "=" * 55)
print("🔍 [Render Pipeline Diagnosis Stages Saved]")
print("=" * 55)
print(f"Stage 0 (Raw VisPy Render) : {raw_render.shape[1]}x{raw_render.shape[0]} -> stage0_raw_render.png")
print(f"Stage 1 (After Anchor)     : {stage1_anchor.shape[1]}x{stage1_anchor.shape[0]} -> stage1_anchor.png")
print(f"Stage 2 (After Border)     : {stage2_border.shape[1]}x{stage2_border.shape[0]} -> stage2_border.png")
print(f"Stage 3 (After CropBorder) : {stage3_final.shape[1]}x{stage3_final.shape[0]} -> stage3_final.png")
print(f"Crop Border Pixels [T,L,B,R]: Top={o_t}px, Left={o_l}px, Bottom={o_b}px, Right={o_r}px")
print("=" * 55 + "\n")