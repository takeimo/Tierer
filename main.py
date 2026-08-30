import sys
import time

# ========== 1. 起動直後に即座にバナーを表示（重いimportの前） ==========
print("\n" + "=" * 60)
print("👑 Tierer [v2026.08.30] - 3D Layered Depth Inpainting")
print("⚡ Initializing AI engines, please wait a moment...")
print("=" * 60 + "\n", flush=True)

start_time = time.time()

# 重いライブラリの読み込み
import copy
from datetime import datetime
import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import imageio.v2 as imageio
from tqdm import tqdm
import yaml
import cv2
import torch

from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
from bilateral_filtering import sparse_bilateral_filtering
from boostmonodepth_utils import run_boostmonodepth

# インペイント用ニューラルネットワークのインポート
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net

# MiDaS による深度推定モジュール
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils

# ========== 2. 引数パーサー（--config を省略可能に設定） ==========
parser = argparse.ArgumentParser(description="Tierer: 3D Layered Depth Inpainting Pipeline")
parser.add_argument(
    "--config",
    type=str,
    default="argument.yml",
    help="Path to the YAML configuration file (default: argument.yml)"
)
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"[Tierer Error] Configuration file '{args.config}' not found.")

if args.config == "argument.yml":
    print(f"[Config] Using default configuration: {args.config}")
else:
    print(f"[Config] Using custom configuration: {args.config}")

config = yaml.safe_load(open(args.config, 'r'))

if config['offscreen_rendering'] is True:
    vispy.use(app='egl')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)

def ensure_checkpoints(config):
    import shutil
    from huggingface_hub import hf_hub_download
    
    repo_id = "takeimo/tierer-models"
    model_mapping = {
        "color-model.pth": config.get("color_model_ckpt", "checkpoints/color-model.pth"),
        "depth-model.pth": config.get("depth_model_ckpt", "checkpoints/depth-model.pth"),
        "edge-model.pth": config.get("edge_model_ckpt", "checkpoints/edge-model.pth"),
        "model.pt": config.get("MiDaS_model_ckpt", "MiDaS/model.pt"),
    }
    for filename, target_path in model_mapping.items():
        if not os.path.exists(target_path):
            print(f"モデルをダウンロード中: {filename} (from {repo_id}) ...")
            target_dir = os.path.dirname(target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy(cached_path, target_path)

ensure_checkpoints(config)

sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'])
normal_canvas, all_canvas = None, None

def resolve_device(cfg):
    gpu_cfg = cfg.get("gpu_ids", "auto")
    
    # 1. ユーザーが明示的に CPU を指定した場合
    if gpu_cfg == -1 or str(gpu_cfg).lower() == "cpu":
        print("[Device] Running on CPU (user specified).")
        return "cpu"
    
    # 2. ユーザーが明示的に GPU番号 (0, 1等) を指定した場合
    if isinstance(gpu_cfg, int) and gpu_cfg >= 0:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(gpu_cfg)
            print(f"[Device] Running on GPU {gpu_cfg}: {gpu_name}")
            return gpu_cfg
        else:
            print(f"[Device Warning] GPU {gpu_cfg} was specified, but CUDA is not available. Falling back to CPU.")
            return "cpu"
            
    # 3. "auto" または未指定の場合: 自動判別
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] NVIDIA GPU detected: {gpu_name}. Running on CUDA.")
        return 0
    else:
        print("[Device] No CUDA GPU detected. Running on CPU.")
        return "cpu"

device = resolve_device(config)

print(f"running on device {device}")

for idx, sample in enumerate(sample_list):
    depth = None
    print(f"\n{'='*20} [{idx+1}/{len(sample_list)}] Processing: {sample['src_pair_name']} {'='*20}")
    mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
    image = imageio.imread(sample['ref_img_fi'])

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running depth extraction ...")
    if config['use_boostmonodepth'] is True:
        run_boostmonodepth(sample['ref_img_fi'], config['src_folder'], config['depth_folder'])
    elif config['require_midas'] is True:
        run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                  config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640, device=device)

    if 'npy' in config['depth_format']:
        config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
    else:
        config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    # 透過PNG (4チャンネル RGBA) の場合は RGB (3チャンネル) に安全変換
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    # グレースケール画像 (2次元) の場合は 3チャンネルに拡張
    elif image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False
    image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
    depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
    mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
    if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
        vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
        depth = vis_depths[-1]
        model = None
        torch.cuda.empty_cache()
        print("Start Running 3D_Photo ...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading edge model ...")
        depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                       map_location=torch.device(device))
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.to(device)
        depth_edge_model.eval()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading depth model ...")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                       map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        depth_feat_model = depth_feat_model.to(device)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading rgb model ...")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                     map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        rgb_model = rgb_model.to(device)
        graph = None


        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building 3D LDI mesh and inpainting ...")
        rt_info = write_ply(image,
                              depth,
                              sample['int_mtx'],
                              mesh_fi,
                              config,
                              rgb_model,
                              depth_edge_model,
                              depth_edge_model,
                              depth_feat_model)

        if rt_info is False:
            continue
        rgb_model = None
        color_feat_model = None
        depth_edge_model = None
        depth_feat_model = None
        torch.cuda.empty_cache()
    if config['save_ply'] is True or config['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rendering video ...")
    videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
    top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
    left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
    down, right = top + config['output_h'], left + config['output_w']
    border = [int(xx) for xx in [top, down, left, right]]
    normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                        copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                        image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                        videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                        mean_loc_depth=mean_loc_depth)

total_time = time.time() - start_time
mins, secs = divmod(int(total_time), 60)
print(f"\n{'='*55}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 All processes completed successfully! (Total time: {mins}m {secs:02d}s)")
print(f"{'='*55}\n")
