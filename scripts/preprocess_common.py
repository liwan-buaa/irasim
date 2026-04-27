import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from decord import VideoReader, cpu

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from compat.hf_hub import ensure_hf_hub_compat

ensure_hf_hub_compat()
from diffusers.models import AutoencoderKL
from tqdm import tqdm


@dataclass
class SplitResult:
    split: str
    total_entries: int
    selected_entries: int
    processed_entries: int
    skipped_entries: int
    latent_generated: int
    action_dim_hist: Dict[str, int]
    annotation_files: List[str]


def parse_args_common(
    description: str,
    default_raw_root: str,
    default_output_root: str,
    default_dataset_name: str,
    default_video_size: Tuple[int, int],
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw-root", type=str, default=default_raw_root)
    parser.add_argument("--output-root", type=str, default=default_output_root)
    parser.add_argument("--dataset-name", type=str, default=default_dataset_name)
    parser.add_argument("--train-json", type=str, default="annotations_train.json")
    parser.add_argument("--val-json", type=str, default="annotations_eval.json")
    parser.add_argument("--max-train-episodes", type=int, default=-1)
    parser.add_argument("--max-val-episodes", type=int, default=-1)
    parser.add_argument(
        "--video-size",
        nargs=2,
        type=int,
        default=[int(default_video_size[0]), int(default_video_size[1])],
        metavar=("H", "W"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--encode-latent", action="store_true", default=True)
    parser.add_argument("--no-encode-latent", dest="encode_latent", action="store_false")
    parser.add_argument("--video-mode", type=str, default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--overwrite-latent", action="store_true", default=False)
    parser.add_argument(
        "--vae-model-path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` was requested but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    abs_src = src.resolve()
    os.symlink(abs_src, dst)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def encode_video_to_latent(
    video_path: Path,
    output_path: Path,
    vae: AutoencoderKL,
    device: torch.device,
    target_hw: Tuple[int, int],
    batch_size: int,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False
    ensure_dir(output_path.parent)

    latents = []
    active_device = device

    def _encode_np_batch(np_batch):
        nonlocal active_device
        frames = torch.from_numpy(np_batch).permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(frames, size=target_hw, mode="bilinear", align_corners=False)
        frames = (frames - 0.5) / 0.5
        try:
            frames = frames.to(active_device)
            latent = vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)
            latents.append(latent.cpu())
        except torch.OutOfMemoryError:
            # Fallback path: if CUDA is busy/fragmented, continue encoding on CPU.
            if active_device.type == "cuda":
                torch.cuda.empty_cache()
                active_device = torch.device("cpu")
                vae.to(active_device)
                frames = frames.to(active_device)
                latent = vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)
                latents.append(latent.cpu())
            else:
                raise

    with torch.no_grad():
        try:
            reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            num_frames = len(reader)
            for start in range(0, num_frames, batch_size):
                end = min(start + batch_size, num_frames)
                frame_ids = list(range(start, end))
                np_batch = reader.get_batch(frame_ids).asnumpy()
                _encode_np_batch(np_batch)
        except Exception:
            reader = imageio.get_reader(str(video_path))
            buffer = []
            for frame in reader:
                buffer.append(frame)
                if len(buffer) == batch_size:
                    _encode_np_batch(np.stack(buffer, axis=0))
                    buffer = []
            reader.close()
            if buffer:
                _encode_np_batch(np.stack(buffer, axis=0))

    latent_tensor = torch.cat(latents, dim=0)
    torch.save(latent_tensor, output_path)
    return True


def parse_task_episode(action_rel_path: str) -> Tuple[str, str]:
    # e.g. "TASK_NAME/123/actions.npy"
    parts = Path(action_rel_path).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected action path format: {action_rel_path}")
    task_name = parts[-3]
    episode_id = parts[-2]
    return task_name, episode_id


def limit_entries(entries: List[dict], max_count: int) -> List[dict]:
    if max_count is None or max_count < 0:
        return entries
    return entries[:max_count]


def make_output_dirs(dataset_root: Path) -> None:
    for rel in [
        "annotation/train",
        "annotation/val",
        "videos/train",
        "videos/val",
        "latent_videos/train",
        "latent_videos/val",
        "actions/train",
        "actions/val",
        "reports",
    ]:
        ensure_dir(dataset_root / rel)


def maybe_build_vae(encode_latent: bool, vae_model_path: str, device: torch.device):
    if not encode_latent:
        return None
    vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    return vae


def summarize_and_save_report(dataset_root: Path, split_results: List[SplitResult], args_dict: dict) -> None:
    report = {
        "args": args_dict,
        "splits": [],
    }
    for r in split_results:
        report["splits"].append(
            {
                "split": r.split,
                "total_entries": r.total_entries,
                "selected_entries": r.selected_entries,
                "processed_entries": r.processed_entries,
                "skipped_entries": r.skipped_entries,
                "latent_generated": r.latent_generated,
                "action_dim_hist": r.action_dim_hist,
                "annotation_files": r.annotation_files,
            }
        )
    save_json(dataset_root / "reports" / "preprocess_report.json", report)


def create_episode_annotation(
    split: str,
    task_name: str,
    episode_id: str,
    camera_order: List[str],
    num_frames: int,
    action_dim: int,
    action_rel_path: str,
    videos_rel: Dict[str, str],
    latents_rel: Dict[str, str],
    source_name: str,
    instructions: List[str],
    raw_entry: dict,
) -> dict:
    return {
        "episode_id": str(episode_id),
        "task_name": str(task_name),
        "split": split,
        "camera_order": list(camera_order),
        "num_frames": int(num_frames),
        "action_dim": int(action_dim),
        "action_path": action_rel_path,
        "videos": {k: {"video_path": v} for k, v in videos_rel.items()},
        "latent_videos": {k: {"latent_video_path": v} for k, v in latents_rel.items()},
        "source": source_name,
        "instructions": instructions,
        "meta": raw_entry.get("meta", {}),
    }
