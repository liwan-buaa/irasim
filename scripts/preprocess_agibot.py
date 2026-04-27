import os
import sys
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_common import (
    SplitResult,
    choose_device,
    create_episode_annotation,
    encode_video_to_latent,
    limit_entries,
    load_json,
    make_output_dirs,
    maybe_build_vae,
    maybe_link_or_copy,
    parse_args_common,
    parse_task_episode,
    save_json,
    summarize_and_save_report,
)


def parse_args():
    return parse_args_common(
        description="Preprocess AgiBot raw data into IRASim-ready format.",
        default_raw_root="/data/liwan/IRASim/work_dirs/Datasets/EWM_infer_meta/agibot",
        default_output_root="work_dirs/data/agibot_irasim",
        default_dataset_name="agibot",
        default_video_size=(192, 256),
    )


def process_split(
    split_name,
    entries,
    max_entries,
    raw_root,
    dataset_root,
    video_mode,
    encode_latent,
    vae,
    device,
    target_hw,
    batch_size,
    overwrite_latent,
):
    selected = limit_entries(entries, max_entries)
    processed_entries = 0
    skipped_entries = 0
    latent_generated = 0
    action_dim_hist = {}
    annotation_files = []
    debug_err_printed = 0

    for entry in tqdm(selected, desc=f"agibot/{split_name}"):
        try:
            action_rel_src = entry["actions"]
            task_name, episode_id = parse_task_episode(action_rel_src)
            camera_order = list(entry["camera_order"])
            instructions = list(entry.get("instructions", []))

            action_src = raw_root / action_rel_src
            action_np = np.load(action_src)
            num_frames = min(int(entry["num_frames"]), int(action_np.shape[0]))
            action_dim = int(action_np.shape[1])
            action_dim_hist[str(action_dim)] = action_dim_hist.get(str(action_dim), 0) + 1

            action_rel_dst = f"actions/{split_name}/{task_name}/{episode_id}/actions.npy"
            action_dst = dataset_root / action_rel_dst
            maybe_link_or_copy(action_src, action_dst, video_mode)

            videos_rel_dst = {}
            latents_rel_dst = {}
            for cam in camera_order:
                video_rel_src = entry["videos"][cam]
                video_src = raw_root / video_rel_src
                video_rel_dst = f"videos/{split_name}/{task_name}/{episode_id}/{cam}.mp4"
                video_dst = dataset_root / video_rel_dst
                maybe_link_or_copy(video_src, video_dst, video_mode)
                videos_rel_dst[cam] = video_rel_dst

                latent_rel_dst = f"latent_videos/{split_name}/{task_name}/{episode_id}/{cam}.pt"
                latent_dst = dataset_root / latent_rel_dst
                latents_rel_dst[cam] = latent_rel_dst
                if encode_latent:
                    created = encode_video_to_latent(
                        video_path=video_src,
                        output_path=latent_dst,
                        vae=vae,
                        device=device,
                        target_hw=target_hw,
                        batch_size=batch_size,
                        overwrite=overwrite_latent,
                    )
                    if created:
                        latent_generated += 1

            ann = create_episode_annotation(
                split=split_name,
                task_name=task_name,
                episode_id=episode_id,
                camera_order=camera_order,
                num_frames=num_frames,
                action_dim=action_dim,
                action_rel_path=action_rel_dst,
                videos_rel=videos_rel_dst,
                latents_rel=latents_rel_dst,
                source_name="agibot",
                instructions=instructions,
                raw_entry=entry,
            )
            ann_name = f"{task_name}__{episode_id}.json"
            ann_path = dataset_root / "annotation" / split_name / ann_name
            save_json(ann_path, ann)
            annotation_files.append(str(Path("annotation") / split_name / ann_name))
            processed_entries += 1
        except Exception:
            skipped_entries += 1
            if debug_err_printed < 3:
                debug_err_printed += 1
                print(f"[WARN] agibot/{split_name} skip entry due to exception:")
                print(traceback.format_exc())

    return SplitResult(
        split=split_name,
        total_entries=len(entries),
        selected_entries=len(selected),
        processed_entries=processed_entries,
        skipped_entries=skipped_entries,
        latent_generated=latent_generated,
        action_dim_hist=action_dim_hist,
        annotation_files=annotation_files,
    )


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    dataset_root = Path(args.output_root) / args.dataset_name
    make_output_dirs(dataset_root)

    device = choose_device(args.device)
    vae = maybe_build_vae(args.encode_latent, args.vae_model_path, device)
    target_hw = (int(args.video_size[0]), int(args.video_size[1]))

    train_entries = load_json(raw_root / args.train_json)
    val_entries = load_json(raw_root / args.val_json)

    split_results = []
    split_results.append(
        process_split(
            split_name="train",
            entries=train_entries,
            max_entries=args.max_train_episodes,
            raw_root=raw_root,
            dataset_root=dataset_root,
            video_mode=args.video_mode,
            encode_latent=args.encode_latent,
            vae=vae,
            device=device,
            target_hw=target_hw,
            batch_size=args.batch_size,
            overwrite_latent=args.overwrite_latent,
        )
    )
    split_results.append(
        process_split(
            split_name="val",
            entries=val_entries,
            max_entries=args.max_val_episodes,
            raw_root=raw_root,
            dataset_root=dataset_root,
            video_mode=args.video_mode,
            encode_latent=args.encode_latent,
            vae=vae,
            device=device,
            target_hw=target_hw,
            batch_size=args.batch_size,
            overwrite_latent=args.overwrite_latent,
        )
    )

    summarize_and_save_report(dataset_root, split_results, vars(args))


if __name__ == "__main__":
    main()
