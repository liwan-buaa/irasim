# IRASim 新数据集接入执行规范（libero / agibot）

更新时间：2026-04-26
状态：可开工（需求已冻结，可直接实施）

## 1. 项目目标
把 `libero` 和 `agibot` 接入 IRASim，达到与现有数据集同级可用性：
1. 单卡/多卡训练可直接运行。
2. 短轨迹评估生成可直接运行。
3. 评估结果按指定目录和视频格式输出。
4. 保留 MP4 与 latent 两套数据资产。
5. 当前阶段不计算任何评估指标，只产出可视化对比结果。

## 2. 已冻结需求（已确认）
1. 必须新增两个独立数据管线：`dataset_libero.py`、`dataset_agibot.py`。
2. 禁止复用 `dataset_3D.py` 的 `state + continuous_gripper_state -> action` 在线反算动作逻辑。
3. 动作由新管线直接读取并输出到 `data['action']`。
4. `agibot` 动作直接使用 16 维。
5. 模型分支输入维度固定：`libero=7`、`agibot=16`。
6. 相机策略：每个视角作为独立数据源；训练对同一 episode 随机选视角。
7. split 规则：
   - `agibot`: `annotations_train.json -> train`, `annotations_eval_small.json -> val`
   - `libero`: `annotations_train.json -> train`, `annotations_eval.json -> val`
8. 不使用 `test` split；训练与评估统一使用 `val`。
9. 必须 `pre_encode=true`，且保留 MP4 + latent（`.pt`）两套。
10. 处理后数据根路径：
   - `work_dirs/data/libero_irasim`
   - `work_dirs/data/agibot_irasim`
11. 当前评估范围仅包含短轨迹评估，不做长轨迹评估。
12. 分辨率约束（训练+推理统一）：
   - `libero` 使用 `128x128`（`H x W = 128 x 128`）
   - `agibot` 使用 `192x256`（`H x W = 192 x 256`）

## 3. 短轨迹评估输出需求（已确认）
1. 未来预测帧数：`libero=81`、`agibot=57`（不含条件起始帧）。
2. 预测方式：同一 `episode + start_frame` 下，各视角独立预测，再拼接。
3. 对比视频：叶子目录只需 `comparison.mp4`。
4. 视频 FPS：30。
5. 拼图布局：
   - `libero`: `2x2`（上排 GT，下排 Pred，列顺序 `[head, hand]`）
   - `agibot`: `2x3`（上排 GT，下排 Pred，列顺序 `[head, hand_left, hand_right]`）
6. 输出根目录：
   - `work_dirs/irasim_infer/libero`
   - `work_dirs/irasim_infer/agibot`
7. 输出层级风格：目录组织方式对齐 `work_dirs/inference/libero90_true`（但根目录使用第 6 条路径）。
8. 短轨迹评估起点与步长固定：每个 episode 从第 `0` 帧开始，之后按 `40` 帧间隔评估一次（`start_frame = 0, 40, 80, ...`）。
9. 尾窗口补齐规则（固定）：
   - 若某个起点下 `GT` 可用未来帧数不足 `pred_len`，`GT` 序列用最后一帧重复补齐到 `pred_len`。
   - 预测结果先在 `GT` 原始尾帧位置截断（仅保留与原始 `GT` 等长部分），再用截断后的最后一帧重复补齐到 `pred_len`。
10. 起点列表生成逻辑固定为你提供的 `_build_start_list(overall_len)` 语义：
   - 默认按 `window_stride=40` 递增取起点；
   - 若 `overall_len < pred_len`，至少保留 `start=0`；
   - 允许追加一个尾部 `next_start`（只要 `< overall_len`）用于覆盖最后一段。
11. 短轨迹输出命名固定：`<out_root>/<task>/<episode_id>_<start_frame>/comparison.mp4`。

## 4. 工程交付物
1. `scripts/preprocess_libero.py`
2. `scripts/preprocess_agibot.py`
3. `dataset/dataset_libero.py`
4. `dataset/dataset_agibot.py`
5. `dataset/__init__.py` 新增分发入口（libero/agibot）
6. 训练配置：
   - `configs/train/libero/frame_ada.yaml`
   - `configs/train/agibot/frame_ada.yaml`
7. 评估配置：
   - `configs/evaluation/libero/frame_ada.yaml`
   - `configs/evaluation/agibot/frame_ada.yaml`
8. 使用文档：训练命令、评估命令、输出目录说明

## 5. 数据契约（落地标准）

### 5.1 处理后目录结构
`work_dirs/data/<dataset>_irasim/` 下至少包含：
1. `annotation/train/`
2. `annotation/val/`
3. `videos/train/`
4. `videos/val/`
5. `latent_videos/train/`
6. `latent_videos/val/`
7. `reports/`（数据质检报告）

补充（与现有数据集一致）：
1. latent 以“整条 episode + camera”为单位存储（不按滑窗拆分）。
2. 样本窗口通过 `start_frame_id` 在 Dataset 中索引切片。

### 5.2 样本返回字段契约（Dataset 输出）
每次 `__getitem__` 返回：
1. `data['action']`：`[num_frames-1, action_dim]`
2. `data['latent']`（`pre_encode=true`）
3. `data['video']`（用于 `return_video=True` 场景）
4. `data['video_name']`：至少包含 `episode_id`, `cam_id`, `start_frame_id`

### 5.3 时序契约
1. 输入条件帧数：`mask_frame_num`（当前默认 1）
2. 动作长度必须对应当前样本窗口（`num_frames-1`）
3. 评估生成未来帧数固定：`libero=81`, `agibot=57`
4. 短轨迹评估窗口起点规则固定：每个 episode 使用 `start_frame = 0, 40, 80, ...`（步长 40）。
5. 短轨迹尾窗口对齐：`GT`/`Pred` 在可用帧不足时均按“最后一帧复制”补齐到目标 `pred_len`，且 `Pred` 需先按真实 `GT` 长度截断后再补齐。
6. 训练窗口规则对齐现有数据集：
   - `num_frames=16`
   - `sequence_interval=1`
   - `train` 起始步长 `start_frame_interval=1`
   - `val` 起始步长 `start_frame_interval=16`
   - 仅保留完整窗口，不做尾部补齐

## 6. 多视角训练实现规则
1. 主干前向仍是单视角张量，不做多视角联合编码。
2. 采样单位定义为 `(episode_id, start_frame_id, camera_id)`。
3. `train`：同一 episode 的可用视角集合中随机取一个视角。
4. `val`：固定视角顺序，确保可复现。
5. 评估视频拼接前先完成各视角独立预测。

## 7. 预处理脚本职责细化

### 7.1 preprocess_libero.py
1. 读取原始 `annotations_train.json` 与 `annotations_eval.json`。
2. 解析多视角视频与动作，并建立 train/val 样本索引。
3. 生成 latent `.pt`（按视角/episode 组织）。
4. 输出质检报告：样本数、缺失、动作维度、长度对齐统计。

### 7.2 preprocess_agibot.py
1. 读取原始 `annotations_train.json` 与 `annotations_eval_small.json`。
2. 使用 16 维动作直接入库，不降维。
3. 解析 3 视角视频并建立 train/val 样本索引。
4. 生成 latent `.pt`。
5. 输出质检报告。

## 8. 训练/评估配置约束
1. `dataset` 分别为 `libero` 与 `agibot`。
2. `pre_encode: true`。
3. `mode: val` 用于评估。
4. `num_frames=16`，并与数据窗口采样规则一致。
5. `sequence_interval=1`，`val_start_frame_interval=16`，训练起始步长为 1。
6. 仅短轨迹评估：评估脚本参数需支持未来帧长度覆写（libero 81 / agibot 57）。
7. 评估脚本需支持独立的短轨迹起点采样策略：固定从 `0` 开始、步长 `40`，不复用训练/验证 DataLoader 的起始步长配置。
8. `video_size` 固定：
   - `libero: [128, 128]`
   - `agibot: [192, 256]`

## 9. 验收标准（DoD）
1. 两个数据集训练都能启动并稳定迭代（单卡/多卡）。
2. `val` 模式评估样本使用全量 `val`。
3. `val` 模式短轨迹评估视频生成成功。
4. 输出目录与命名符合约定。
5. `comparison.mp4`：
   - FPS=30
   - libero 为 2x2
   - agibot 为 2x3
   - 上 GT 下 Pred
   - 仅保存 `comparison.mp4`，不输出其他评估视频
6. 不计算任何评估指标（Latent L2 / PSNR / SSIM / FID / FVD 均不执行）。
7. 数据质检报告无阻断错误（缺失/错维/长度异常在阈值内）。

## 10. 执行顺序
1. 完成两套预处理脚本。
2. 先仅生成小规模子集数据用于联调测试（MP4 + latent），避免内存压力。
3. 生成方式通过参数可扩展到全量数据。
4. 完成两套 Dataset。
5. 注册入口与配置。
6. 跑通训练。
7. 跑通短轨迹评估生成。
8. 固化最终文档。

## 11. 当前仍待你补充
目前无阻塞项。相机字段名映射已确认（train/eval 一致）：
1. `libero`：`head`, `hand`
2. `agibot`：`head`, `hand_left`, `hand_right`

## 12. 文档维护约定
按你的要求：每轮需求对话后更新本文件，作为唯一执行基线。

## 13. 变更约束（新增）
1. 不得删除任何现有文件（包括代码、配置、数据、文档）。
2. 所有新增/修改都必须同步记录到 `change.md`。
3. `change.md` 作为变更台账，至少包含：日期、文件路径、改动摘要、改动原因。

## 14. 执行实测（2026-04-26）
1. 已按最新约束重跑小规模预处理：
   - libero: train=2, val=2，处理成功（动作维度 7，latent 对齐 `128x128 -> 16x16`）。
   - agibot: train=2, val=2，处理成功（动作维度 16，latent 对齐 `192x256 -> 24x32`）。
2. 已验证 Dataset 取样输出契约：
   - `libero`: `action=(15,7)`, `latent=(16,4,16,16)`
   - `agibot`: `action=(15,16)`, `latent=(16,4,24,32)`
3. 环境问题与处理已写入 `change.md`（`diffusers/hf_hub` 兼容、AV1 解码回退、OOM 回退）。
