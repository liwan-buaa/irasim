# Change Log

更新时间：2026-04-26
规则：不得删除任何文件；所有改动必须在本文件登记。

## 2026-04-26

### 1) models/irasim.py
- 改动摘要：
  - 新增 `libero` 与 `agibot` 的模型分支适配（`extras==3` 与 `extras==5` 路径）。
  - 固定输入维度：`libero=7`，`agibot=16`。
  - 补充不支持数据集时的显式异常。
- 改动原因：
  - 按需求为两个新数据集提供独立模型侧适配，不复用原有 3D 数据集动作推导逻辑。

### 2) try.py
- 改动摘要：
  - 将 `.npy` 查看逻辑改为 `numpy.load`。
  - 增加命令行参数（路径、输出行数、是否全量打印）用于快速检查数组内容。
- 改动原因：
  - 满足“查看 npy 文件里存的是什么”的调试需求。

### 3) process_data.md
- 改动摘要：
  - 新增并完善了 `libero/agibot` 数据处理说明，覆盖数据组织、处理流程和使用方式。
- 改动原因：
  - 按需求提供可执行的数据处理文档说明。

### 4) agent.md
- 改动摘要：
  - 持续同步并冻结工程需求：数据管线、动作维度、多视角策略、split 规则、`pre_encode`、短轨迹评估、输出目录与命名。
  - 明确短轨迹起点采样：每个 episode 从 `0` 开始，步长 `40`，并采用 `_build_start_list` 语义。
  - 明确尾窗口补齐规则：GT 与 Pred 在尾部不足时按最后一帧补齐到 `pred_len`。
  - 明确相机键名映射：`libero(head, hand)`、`agibot(head, hand_left, hand_right)`。
  - 新增“不得删除文件 + 所有改动登记 change.md”约束。
- 改动原因：
  - 将多轮需求收敛为唯一、可执行的工程基线文档。

### 5) agent.md
- 改动摘要：
  - 新增分辨率硬约束：`libero` 训练/推理统一 `128x128`，`agibot` 训练/推理统一 `192x256`。
  - 在“训练/评估配置约束”中补充固定 `video_size`：
    - `libero: [128, 128]`
    - `agibot: [192, 256]`
- 改动原因：
  - 按最新需求锁定两套数据集的统一输入分辨率，避免训练与推理分辨率不一致。

### 6) scripts/preprocess_common.py
- 改动摘要：
  - 新增 preprocess 公共模块，统一参数解析、目录构建、annotation 生成、latent 编码与报告输出。
  - 新增视频解码回退逻辑：`decord` 失败时自动使用 `imageio(ffmpeg)`。
  - 新增显存保护逻辑：`CUDA OOM` 时自动回退到 CPU 编码，避免整条样本被跳过。
  - 调整默认 VAE 路径为 `stabilityai/stable-diffusion-xl-base-1.0`。
- 改动原因：
  - 统一两套预处理脚本实现并增强鲁棒性，解决 agibot 视频编码兼容与显存波动问题。

### 7) scripts/preprocess_libero.py / scripts/preprocess_agibot.py
- 改动摘要：
  - 新增两套独立预处理入口脚本，按 split 读取原始 annotations，输出 IRASim 可用格式。
  - `libero` 默认处理分辨率设为 `128x128`，`agibot` 默认处理分辨率设为 `192x256`。
  - 增加异常打印（前几条），便于快速定位失败原因。
- 改动原因：
  - 满足“独立预处理管线 + 分辨率冻结 + 小样本联调可执行”的工程要求。

### 8) dataset/dataset_multiview_action.py
- 改动摘要：
  - 新增多视角动作数据集基类，直接读取 `actions.npy`，不复用 `dataset_3D` 动作反算逻辑。
  - 支持 train 随机视角与 val 固定视角顺序。
  - 新增短轨迹评估辅助接口（按起点取动作窗、读取起始 latent、读取 GT 未来帧）。
  - 视频读取增加 `decord -> imageio` 回退，避免 AV1 视频导致读取失败。
- 改动原因：
  - 为 libero/agibot 提供统一、可复用且与需求一致的 Dataset 基础能力。

### 9) dataset/dataset_libero.py / dataset/dataset_agibot.py / dataset/__init__.py
- 改动摘要：
  - 新增 `Dataset_Libero`（动作维 7）和 `Dataset_Agibot`（动作维 16）。
  - 在 `dataset/__init__.py` 中注册 `libero/agibot` 分发入口。
- 改动原因：
  - 满足新增数据集入口要求，并保证训练/评估主流程可直接识别两个新数据集。

### 10) evaluate/generate_multiview_short_video.py / main.py
- 改动摘要：
  - 新增 `libero/agibot` 专用短轨迹评估生成器：
    - 起点采样按 `_build_start_list`（步长 40）
    - 多视角独立预测后拼图
    - 尾窗口 GT/Pred 补齐规则落地
    - 仅输出 `comparison.mp4`（FPS=30）
  - 在 `main.py` 的 `do_evaluate` 分支中为 `libero/agibot` 接入该评估器。
- 改动原因：
  - 使评估输出路径、命名与视频布局严格符合冻结需求。

### 11) configs/train/* / configs/evaluation/*
- 改动摘要：
  - 新增：
    - `configs/train/libero/frame_ada.yaml`
    - `configs/train/agibot/frame_ada.yaml`
    - `configs/evaluation/libero/frame_ada.yaml`
    - `configs/evaluation/agibot/frame_ada.yaml`
  - 分辨率已按最新约束固定：
    - libero `video_size: [128, 128]`
    - agibot `video_size: [192, 256]`
  - `pre_encode: true`、短轨迹参数（pred_len/stride/fps/camera_order/output_root）已配置。
- 改动原因：
  - 提供可直接执行的训练/评估配置模板。

### 12) compat/hf_hub.py / util.py / evaluate/* / main.py
- 改动摘要：
  - 新增 `hf_hub` 兼容补丁模块，并在入口位置注入，兼容旧 diffusers 对新 huggingface_hub API 的调用差异。
  - `util.py` 新增补丁注入，避免被独立导入时再次报错。
- 改动原因：
  - 解决当前环境中的 `diffusers` 与 `huggingface_hub` 版本不兼容问题。

### 13) docs/libero_agibot_usage.md
- 改动摘要：
  - 新增可执行命令文档：小样本预处理、全量预处理、训练与短轨迹评估命令。
- 改动原因：
  - 满足工程交付中的“使用文档”要求，降低复现成本。

### 14) agent.md
- 改动摘要：
  - 新增“执行实测”章节，记录小规模预处理成功结果、Dataset 输出形状验证结果以及环境问题处理状态。
- 改动原因：
  - 按要求将最新执行状态持续同步到基线文档，保证实现与文档一致。

## 环境问题记录（本轮执行中确认）

1) `diffusers` 与 `huggingface_hub` 版本兼容问题  
- 现象：
  - `hf_cache_home` / `HfFolder` / `cached_download` / `use_auth_token` 相关导入或参数错误。
- 处理：
  - 增加 `compat/hf_hub.py` 兼容层，并在 `main.py`、`util.py`、`evaluate/*`、`scripts/preprocess_common.py` 注入。

2) agibot 部分 MP4（AV1 编码）与 decord 不兼容  
- 现象：
  - `decord` 报错：`cannot find video stream with wanted index: -1`。
- 处理：
  - 读取路径增加 `decord -> imageio(ffmpeg)` 自动回退。

3) 预处理阶段 GPU 显存不足（受其他进程占用影响）  
- 现象：
  - VAE 编码出现 `torch.OutOfMemoryError`，导致样本被跳过。
- 处理：
  - 编码函数加入 `CUDA OOM -> CPU` 自动回退，保证流程继续并完成样本处理。
