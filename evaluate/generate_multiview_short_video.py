import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from compat.hf_hub import ensure_hf_hub_compat
ensure_hf_hub_compat()
from diffusers.schedulers import DDPMScheduler, PNDMScheduler
from tqdm import tqdm

from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline


def _build_start_list(overall_len, pred_len, window_stride):
    if overall_len <= 0:
        return []
    max_start = overall_len - pred_len
    if max_start >= 0:
        starts = list(range(0, max_start + 1, window_stride))
    else:
        starts = [0]
    next_start = starts[-1] + window_stride
    if next_start < overall_len:
        starts.append(next_start)
    return starts


def _resize_uint8_frames(frames_uint8, target_hw):
    if frames_uint8.shape[0] == 0:
        return frames_uint8
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def _videos_to_uint8(videos):
    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return videos.permute(0, 2, 3, 1).cpu().contiguous().numpy()


def _predict_future_autoreg(args, pipeline, start_latent, action_seq, device):
    chunk = args.num_frames - 1
    if action_seq.shape[0] == 0:
        return torch.empty((0, 3, args.video_size[0], args.video_size[1]), dtype=torch.float32)

    generated = []
    current_start = start_latent.clone().float().cpu()
    offset = 0
    while offset < action_seq.shape[0]:
        chunk_actions = action_seq[offset : offset + chunk]
        true_len = chunk_actions.shape[0]
        if true_len < chunk:
            pad = chunk_actions[-1:].repeat(chunk - true_len, 1)
            chunk_actions = torch.cat([chunk_actions, pad], dim=0)
        mask_x = current_start.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        actions = chunk_actions.unsqueeze(0).to(device=device, dtype=torch.float32)

        output_type = "both" if args.model != "VDM" else "video"
        pred_videos, pred_latents = pipeline(
            actions,
            mask_x=mask_x,
            video_length=args.num_frames,
            height=args.video_size[0],
            width=args.video_size[1],
            num_inference_steps=args.infer_num_sampling_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            return_dict=False,
            output_type=output_type,
        )

        pred_videos = pred_videos.squeeze(0).detach().cpu()
        generated.append(pred_videos[1 : 1 + true_len])
        if args.model == "VDM":
            current_start = pred_videos[true_len].detach().cpu()
        else:
            pred_latents = pred_latents.squeeze(0).detach().cpu()
            current_start = pred_latents[true_len]
        offset += true_len
    return torch.cat(generated, dim=0)


def _get_scheduler(args):
    if args.sample_method == "PNDM":
        return PNDMScheduler.from_pretrained(
            args.scheduler_path,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    if args.sample_method == "DDPM":
        return DDPMScheduler.from_pretrained(
            args.scheduler_path,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    raise NotImplementedError(args.sample_method)


def _default_camera_order(dataset):
    if dataset == "libero":
        return ["head", "hand"]
    if dataset == "agibot":
        return ["head", "hand_left", "hand_right"]
    raise NotImplementedError(dataset)


def generate_multiview_short_videos(args, val_dataset, model, vae, device, rank=0, world_size=1):
    scheduler = _get_scheduler(args)
    pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    pred_len = int(args.short_eval_pred_len)
    window_stride = int(args.short_eval_window_stride)
    fps = int(args.short_eval_fps)
    out_root = args.short_eval_output_root
    target_hw = (int(args.video_size[0]), int(args.video_size[1]))

    wanted_order = list(getattr(args, "short_eval_camera_order", _default_camera_order(args.dataset)))
    ann_files = sorted(val_dataset.ann_files)

    for ann_idx, ann_file in tqdm(
        list(enumerate(ann_files)),
        total=len(ann_files),
        desc=f"short-eval-{args.dataset}-rank{rank}",
    ):
        if ann_idx % world_size != rank:
            continue
        label = val_dataset.load_annotation(ann_file)
        episode_id = str(label["episode_id"])
        task_name = str(label["task_name"])

        camera_order = list(label["camera_order"])
        cam_order = [c for c in wanted_order if c in camera_order]
        if len(cam_order) != len(wanted_order):
            missing = [c for c in wanted_order if c not in cam_order]
            raise ValueError(f"Missing cameras in annotation: {missing}; got {camera_order}")

        total_frames = int(val_dataset.get_episode_length(label))
        overall_len = total_frames - 1  # number of future frames from start=0
        starts = _build_start_list(overall_len=overall_len, pred_len=pred_len, window_stride=window_stride)

        for start in starts:
            gt_valid_len = min(pred_len, overall_len - start)
            if gt_valid_len <= 0:
                continue

            per_cam_gt = {}
            per_cam_pred = {}
            for cam_key in cam_order:
                action_seq = val_dataset.get_action_window(label, start_frame=start, length=gt_valid_len).float()
                start_latent = val_dataset.get_latent_frame(label, cam_key=cam_key, frame_id=start)
                pred_video = _predict_future_autoreg(args, pipeline, start_latent, action_seq, device)
                pred_video = _videos_to_uint8(pred_video)
                pred_video = _resize_uint8_frames(pred_video, target_hw)

                gt_video = val_dataset.get_video_future_uint8(
                    label, cam_key=cam_key, start_frame=start, future_len=gt_valid_len
                )
                gt_video = _resize_uint8_frames(gt_video, target_hw)

                # Tail padding rule (fixed):
                # 1) GT pads with its last valid GT frame.
                # 2) Pred first truncates to original GT length, then pads with its own tail.
                if gt_video.shape[0] < pred_len:
                    pad = np.repeat(gt_video[-1:], pred_len - gt_video.shape[0], axis=0)
                    gt_video = np.concatenate([gt_video, pad], axis=0)

                pred_video = pred_video[:gt_valid_len]
                if pred_video.shape[0] == 0:
                    pred_video = np.repeat(gt_video[:1], pred_len, axis=0)
                elif pred_video.shape[0] < pred_len:
                    pad = np.repeat(pred_video[-1:], pred_len - pred_video.shape[0], axis=0)
                    pred_video = np.concatenate([pred_video, pad], axis=0)

                per_cam_gt[cam_key] = gt_video
                per_cam_pred[cam_key] = pred_video

            out_dir = os.path.join(out_root, task_name, f"{episode_id}_{start}")
            os.makedirs(out_dir, exist_ok=True)
            out_mp4 = os.path.join(out_dir, "comparison.mp4")

            writer = imageio.get_writer(out_mp4, fps=fps)
            for t in range(pred_len):
                top = np.concatenate([per_cam_gt[c][t] for c in cam_order], axis=1)
                bottom = np.concatenate([per_cam_pred[c][t] for c in cam_order], axis=1)
                frame = np.concatenate([top, bottom], axis=0)
                writer.append_data(frame)
            writer.close()
