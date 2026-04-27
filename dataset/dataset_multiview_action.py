import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torchvision.transforms as T
import imageio.v2 as imageio
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from dataset.video_transforms import Resize_Preprocess, ToTensorVideo


class MultiViewActionDataset(Dataset):
    def __init__(self, args, mode: str, dataset_name: str, action_dim: int):
        super().__init__()
        self.args = args
        self.mode = mode
        self.dataset_name = dataset_name
        self.action_dim = int(action_dim)
        self.training = mode == "train"

        if mode == "train":
            self.data_path = args.train_annotation_path
            self.start_frame_interval = 1
        elif mode == "val":
            self.data_path = args.val_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        elif mode == "test":
            self.data_path = args.test_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.video_path = args.video_path
        self.sequence_interval = args.sequence_interval
        self.sequence_length = args.num_frames
        self.seq_len = args.num_frames - 1
        self.c_act_scaler = np.ones((self.action_dim,), dtype=float)

        self.ann_files = self._init_anns(self.data_path)
        self.samples = self._init_sequences(self.ann_files)
        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if args.debug and not args.do_evaluate:
            self.samples = self.samples[:10]

        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")

        self.error_num = 0
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(args.video_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(args.video_size)),
            ]
        )

    def _init_anns(self, data_dir):
        if not os.path.isdir(data_dir):
            return []
        return sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        )

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_ann = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in as_completed(future_to_ann):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            return samples

        n_frames = int(ann["num_frames"])
        for frame_i in range(0, n_frames, self.start_frame_interval):
            frame_ids = []
            curr = frame_i
            while True:
                if curr > (n_frames - 1):
                    break
                frame_ids.append(curr)
                if len(frame_ids) == self.sequence_length:
                    break
                curr += self.sequence_interval
            if len(frame_ids) == self.sequence_length:
                samples.append({"ann_file": ann_file, "frame_ids": frame_ids})
        return samples

    def __len__(self):
        return len(self.samples)

    def load_annotation(self, ann_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_video(self, video_path, frame_ids):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            assert (np.array(frame_ids) < len(vr)).all()
            assert (np.array(frame_ids) >= 0).all()
            vr.seek(0)
            frame_data = vr.get_batch(frame_ids).asnumpy()
            return frame_data
        except Exception:
            reader = imageio.get_reader(video_path)
            frames = [reader.get_data(int(fid)) for fid in frame_ids]
            reader.close()
            return np.stack(frames, axis=0)

    def _load_tokenized_video(self, video_path, frame_ids):
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file, map_location="cpu")
        assert (np.array(frame_ids) < video_tensor.size(0)).all()
        assert (np.array(frame_ids) >= 0).all()
        return video_tensor[frame_ids]

    def _resolve_cam_key(self, label, cam_id=None, sample_index=None):
        camera_order = list(label["camera_order"])
        if cam_id is None:
            if self.mode == "train":
                return random.choice(camera_order)
            # validation/eval: deterministic fixed order
            if sample_index is None:
                return camera_order[0]
            return camera_order[int(sample_index) % len(camera_order)]
        if isinstance(cam_id, int):
            return camera_order[cam_id % len(camera_order)]
        cam_key = str(cam_id)
        if cam_key not in camera_order:
            raise KeyError(f"Unknown camera key: {cam_key}, expected one of {camera_order}")
        return cam_key

    def _get_frames(self, label, frame_ids, cam_key, pre_encode):
        if pre_encode:
            video_rel = label["latent_videos"][cam_key]["latent_video_path"]
            video_path = os.path.join(self.video_path, video_rel)
            return self._load_tokenized_video(video_path, frame_ids)

        video_rel = label["videos"][cam_key]["video_path"]
        video_path = os.path.join(self.video_path, video_rel)
        frames = self._load_video(video_path, frame_ids)
        frames = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2)
        if self.args.normalize:
            frames = self.preprocess(frames)
        else:
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _load_actions(self, label, frame_ids):
        action_rel = label["action_path"]
        action_path = os.path.join(self.video_path, action_rel)
        all_actions = np.load(action_path)
        actions = all_actions[frame_ids[:-1]]
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"{self.dataset_name} action dim mismatch, expected {self.action_dim}, got {actions.shape[-1]}"
            )
        actions = actions * self.c_act_scaler
        return torch.from_numpy(actions).float()

    def get_episode_length(self, label):
        action_rel = label["action_path"]
        action_path = os.path.join(self.video_path, action_rel)
        action_len = int(np.load(action_path, mmap_mode="r").shape[0])
        return min(int(label["num_frames"]), action_len)

    def get_action_window(self, label, start_frame, length):
        action_rel = label["action_path"]
        action_path = os.path.join(self.video_path, action_rel)
        all_actions = np.load(action_path)
        end = min(start_frame + length, all_actions.shape[0])
        actions = all_actions[start_frame:end]
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"{self.dataset_name} action dim mismatch, expected {self.action_dim}, got {actions.shape[-1]}"
            )
        return torch.from_numpy(actions.astype(np.float32))

    def get_latent_frame(self, label, cam_key, frame_id):
        latent_rel = label["latent_videos"][cam_key]["latent_video_path"]
        latent_path = os.path.join(self.video_path, latent_rel)
        latent = torch.load(latent_path, map_location="cpu")
        return latent[frame_id]

    def get_video_future_uint8(self, label, cam_key, start_frame, future_len):
        video_rel = label["videos"][cam_key]["video_path"]
        video_path = os.path.join(self.video_path, video_rel)
        if future_len <= 0:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        frame_ids = list(range(start_frame + 1, start_frame + 1 + future_len))
        frames = self._load_video(video_path, frame_ids).astype(np.uint8)
        return frames

    def __getitem__(self, index, cam_id=None, return_video=False):
        if not self.training:
            np.random.seed(index)
            random.seed(index)
        try:
            sample = self.samples[index]
            label = self.load_annotation(sample["ann_file"])
            frame_ids = sample["frame_ids"]
            cam_key = self._resolve_cam_key(label, cam_id=cam_id, sample_index=index)

            data = {}
            data["action"] = self._load_actions(label, frame_ids)
            if self.args.pre_encode:
                latent = self._get_frames(label, frame_ids, cam_key=cam_key, pre_encode=True)
                data["latent"] = latent.float()
                if return_video:
                    video = self._get_frames(label, frame_ids, cam_key=cam_key, pre_encode=False)
                    data["video"] = video.float()
            else:
                video = self._get_frames(label, frame_ids, cam_key=cam_key, pre_encode=False)
                data["video"] = video.float()

            data["video_name"] = {
                "task_name": str(label["task_name"]),
                "episode_id": str(label["episode_id"]),
                "start_frame_id": str(frame_ids[0]),
                "cam_id": str(cam_key),
            }
            return data
        except Exception:
            warnings.warn(f"Invalid data encountered: {self.samples[index]}. Skipped.")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.error_num += 1
            return self[np.random.randint(len(self.samples))]
