"""
Download Depth Anything V2 model checkpoint và source code.
"""

import os
import sys
import subprocess


MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
}

REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"


def download_file(url, dest):
    """Tải file từ URL."""
    print(f"Downloading: {url}")
    print(f"  -> {dest}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "requests"
    ])
    import requests
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"\r  Progress: {pct}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="", flush=True)
    print("\n  Done!")


def clone_depth_anything_v2():
    """Clone Depth Anything V2 repo để lấy model code."""
    if os.path.exists("depth_anything_v2"):
        print("[OK] depth_anything_v2/ already exists")
        return

    print("Cloning Depth Anything V2 repository...")
    # Clone sparse - chỉ lấy folder model code
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, "_depth_anything_v2_repo"])

    # Copy chỉ folder depth_anything_v2 (model code)
    import shutil
    src = os.path.join("_depth_anything_v2_repo", "depth_anything_v2")
    if os.path.exists(src):
        shutil.copytree(src, "depth_anything_v2")
        print("[OK] Copied depth_anything_v2/ model code")
    else:
        print("[ERROR] Cannot find depth_anything_v2/ in cloned repo")
        sys.exit(1)

    # Cleanup
    shutil.rmtree("_depth_anything_v2_repo", ignore_errors=True)
    print("[OK] Cleaned up temp repo")


def main():
    print("=" * 50)
    print("  Depth Anything V2 - Model Downloader")
    print("=" * 50)

    # 1. Clone model code
    clone_depth_anything_v2()

    # 2. Download checkpoint
    os.makedirs("checkpoints", exist_ok=True)

    # Mặc định tải ViT-S (nhẹ nhất, phù hợp Jetson Nano)
    encoder = "vits"
    if len(sys.argv) > 1 and sys.argv[1] in MODEL_URLS:
        encoder = sys.argv[1]

    ckpt_name = f"depth_anything_v2_{encoder}.pth"
    ckpt_path = os.path.join("checkpoints", ckpt_name)

    if os.path.exists(ckpt_path):
        print(f"[OK] Checkpoint already exists: {ckpt_path}")
    else:
        url = MODEL_URLS[encoder]
        download_file(url, ckpt_path)

    print("\n" + "=" * 50)
    print(f"  Setup complete!")
    print(f"  Model: {encoder}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Run: python3 main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
