import os
from huggingface_hub import hf_hub_download

# 元論文の公開モデルがミラーリングされているHugging Faceリポジトリ
REPO_ID = "ai-minamo/3d-photo-inpainting"

# 保存先フォルダの作成
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("MiDaS", exist_ok=True)

models = {
    "color-model.pth": "checkpoints/color-model.pth",
    "depth-model.pth": "checkpoints/depth-model.pth",
    "edge-model.pth": "checkpoints/edge-model.pth",
    "model.pt": "MiDaS/model.pt",
}

print("=== 必要なAIモデルをダウンロード中 (初回のみ) ===")
for filename, target_path in models.items():
    if not os.path.exists(target_path):
        print(f"ダウンロード中: {filename} -> {target_path}")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
    else:
        print(f"すでに存在します: {target_path}")

print("=== すべてのモデルの準備が完了しました ===")