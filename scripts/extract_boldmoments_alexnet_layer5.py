"""Extract compact AlexNet layer-5 features for BOLD Moments videos.

The official Algonauts 2021 devkit extracts all AlexNet layers and depends on
torchvision, cv2, and decord. This script keeps the same baseline spirit but is
lighter for the current Mac environment:

- reads BOLD Moments/Algonauts mp4s with imageio-ffmpeg
- uses the official devkit AlexNet architecture and ImageNet checkpoint
- extracts only layer_5
- writes the devkit-compatible PCA files:
  ``alexnet/pca_100/train_layer_5.npy`` and ``test_layer_5.npy``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEFAULT_VIDEO_DIR = Path("data/raw/algonauts2021/AlgonautsVideos268_All_30fpsmax")
DEFAULT_SAVE_DIR = Path("data/raw/algonauts2021/alexnet")
DEFAULT_DEVKIT_ALEXNET = Path(
    "data/external/Algonauts2021_devkit/feature_extraction/alexnet.py"
)
CHECKPOINT_URL = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
CHECKPOINT_KEYS = [
    "conv1.0.weight",
    "conv1.0.bias",
    "conv2.0.weight",
    "conv2.0.bias",
    "conv3.0.weight",
    "conv3.0.bias",
    "conv4.0.weight",
    "conv4.0.bias",
    "conv5.0.weight",
    "conv5.0.bias",
    "fc6.1.weight",
    "fc6.1.bias",
    "fc7.1.weight",
    "fc7.1.bias",
    "fc8.1.weight",
    "fc8.1.bias",
]


def _load_devkit_alexnet(devkit_alexnet: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "algonauts_devkit_alexnet", devkit_alexnet
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AlexNet module from {devkit_alexnet}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.alexnet()


def _download_checkpoint(path: Path) -> None:
    if path.exists() and path.stat().st_size > 100_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[alexnet] downloading checkpoint -> {path}")
    urllib.request.urlretrieve(CHECKPOINT_URL, path)


def _load_model(
    devkit_alexnet: Path, checkpoint: Path, device: torch.device
) -> torch.nn.Module:
    _download_checkpoint(checkpoint)
    model = _load_devkit_alexnet(devkit_alexnet)
    # Official PyTorch AlexNet weights are in the legacy tar format.
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        CHECKPOINT_KEYS[i]: weight
        for i, weight in enumerate(checkpoint_payload.values())
        if i < len(CHECKPOINT_KEYS)
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _video_paths(video_dir: Path, limit: int | None) -> list[Path]:
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 videos found in {video_dir}")
    return videos[:limit] if limit else videos


def _sample_frames(path: Path, n_frames: int) -> list[np.ndarray]:
    reader: Any = imageio.get_reader(str(path), "ffmpeg")  # type: ignore[arg-type]
    try:
        total = reader.count_frames()
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        return [reader.get_data(int(i)) for i in indices]
    finally:
        reader.close()


def _preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
    processed = []
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    for frame in frames:
        image = (
            Image.fromarray(frame)
            .convert("RGB")
            .resize((224, 224), Image.Resampling.BILINEAR)
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        processed.append(np.transpose(arr, (2, 0, 1)))
    return torch.from_numpy(np.stack(processed)).float()


@torch.inference_mode()
def _extract_layer5(
    model: torch.nn.Module,
    video: Path,
    *,
    n_frames: int,
    device: torch.device,
) -> np.ndarray:
    frames = _sample_frames(video, n_frames)
    batch = _preprocess_frames(frames).to(device)
    outputs = model(batch)
    layer5 = outputs[4].detach().cpu().numpy()
    return np.asarray(
        layer5.reshape(layer5.shape[0], -1).mean(axis=0), dtype=np.float32
    )


def _write_pca(
    features: np.ndarray,
    save_dir: Path,
    *,
    n_train: int,
    n_components: int,
) -> dict[str, Any]:
    n_train = min(n_train, features.shape[0])
    n_test = features.shape[0] - n_train
    n_components_eff = min(n_components, n_train, features.shape[1])

    train_x = features[:n_train]
    test_x = features[n_train:]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = (
        scaler.transform(test_x) if n_test else np.empty((0, features.shape[1]))
    )

    pca = PCA(n_components=n_components_eff, random_state=42)
    train_pca = pca.fit_transform(train_scaled).astype(np.float32)
    test_pca = (
        pca.transform(test_scaled).astype(np.float32)
        if n_test
        else np.empty((0, n_components_eff), dtype=np.float32)
    )

    pca_dir = save_dir / "pca_100"
    pca_dir.mkdir(parents=True, exist_ok=True)
    np.save(pca_dir / "train_layer_5.npy", train_pca)
    np.save(pca_dir / "test_layer_5.npy", test_pca)
    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "feature_dim": int(features.shape[1]),
        "n_components": int(n_components_eff),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--devkit-alexnet", type=Path, default=DEFAULT_DEVKIT_ALEXNET)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/raw/algonauts2021/alexnet.pth")
    )
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_train = args.save_dir / "pca_100" / "train_layer_5.npy"
    out_test = args.save_dir / "pca_100" / "test_layer_5.npy"
    if out_train.exists() and out_test.exists() and not args.force:
        print(
            f"[alexnet] outputs already exist under {out_train.parent}; use --force to overwrite"
        )
        return

    device = _choose_device(args.device)
    videos = _video_paths(args.video_dir, args.limit)
    print(
        f"[alexnet] videos={len(videos)} device={device} frames/video={args.n_frames}"
    )
    model = _load_model(args.devkit_alexnet, args.checkpoint, device)

    features = []
    video_ids = []
    for video in tqdm(videos, desc="[alexnet] layer_5"):
        features.append(
            _extract_layer5(model, video, n_frames=args.n_frames, device=device)
        )
        video_ids.append(video.stem)
    feature_array = np.stack(features).astype(np.float32)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.save_dir / "layer_5_all.npy", feature_array)
    (args.save_dir / "video_ids.json").write_text(json.dumps(video_ids, indent=2))
    summary = _write_pca(
        feature_array,
        args.save_dir,
        n_train=args.n_train,
        n_components=args.n_components,
    )
    summary |= {
        "video_dir": str(args.video_dir),
        "save_dir": str(args.save_dir),
        "n_videos": int(len(videos)),
        "device": str(device),
        "n_frames": int(args.n_frames),
    }
    (args.save_dir / "layer_5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[alexnet] wrote {args.save_dir / 'pca_100' / 'train_layer_5.npy'}")
    print(f"[alexnet] wrote {args.save_dir / 'pca_100' / 'test_layer_5.npy'}")


if __name__ == "__main__":
    main()
