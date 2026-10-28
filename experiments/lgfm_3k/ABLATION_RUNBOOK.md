# LG-FM 消融实验运行手册

本手册用于复现 LG-FM 主干的 B1--B3 融合消融。它们都以已完成的、无动态图 LG-FM 主干为参照，固定 batch size、训练轮数、随机种子、数据划分、优化器和评测协议；每一组从随机初始化开始训练，不继承参照模型的权重或 Adam 状态。

| 编号 | Part → Global | Global → Part | 回答的问题 |
| --- | ---: | ---: | --- |
| B1 | 关闭 | 关闭 | 跨分支注意力是否带来收益？ |
| B2 | 关闭 | 开启 | 全局语义向局部部位传播是否有效？ |
| B3 | 开启 | 关闭 | 局部部位信息汇聚到全局是否有效、是否稳定？ |
| 主干参照 | 开启 | 开启 | 双向融合的已完成无动态图 LG-FM 主干 |

`use_local_global_fusion` 是旧配置的总开关。新加入的 `use_part_to_global_fusion` 与 `use_global_to_part_fusion` 分别控制两条注意力支路；旧配置未指定新字段时，仍保持原来的双向行为。关闭支路时，对应的 `MultiheadAttention` 不会被创建，参数量和优化器状态不会混入该消融。

## 1. 代码与数据边界

代码通过 Git 同步；不要把 `outputs_amass/`、数据集、检查点、缓存或日志提交到仓库。训练所需但不随仓库分发的资产如下：

```text
datasets/annotations/frankenstein-dataset/
datasets/annotations/frankenstein-dataset/text_embeddings/
datasets/motions/AMASS_20.0_fps_nh_smplrifke/
pretrained/eval_model/
```

这些目录应来自有权限的数据与预训练模型副本，并保持与主干实验相同的版本。新机器上优先将它们挂载或建立软链接到上述相对路径；不要在 B1--B3 间重复复制。

## 2. 放到服务器

以下命令不包含服务器地址、账号或密码。请使用团队已配置的 SSH alias，或由管理员提供的访问方式。

```bash
ssh <server-alias>
mkdir -p /data/<user>/work
cd /data/<user>/work
git clone https://github.com/changkong11/FrankenMotion.git
cd FrankenMotion
git checkout master
git pull --ff-only origin master
git log -1 --oneline
```

已经有仓库时只需更新代码：

```bash
cd /data/<user>/work/FrankenMotion
git pull --ff-only origin master
git rev-parse --short HEAD       # 应包含 508b96f 或其后续提交
```

若数据存放在共享盘，建议使用软链接而不是复制：

```bash
ln -s /shared/frankenstein/annotations datasets/annotations
ln -s /shared/frankenstein/motions datasets/motions
ln -s /shared/frankenstein/pretrained pretrained
```

执行前先用 `ls` 确认链接目标存在；已有同名真实目录时不要覆盖。

## 3. 高效搭建环境

`scripts/setup_school_environment.sh` 支持通过环境变量指定环境和缓存位置。将环境、pip 缓存、Hugging Face 缓存放在持久磁盘，之后同一服务器上的后续实验无需重复下载或安装。

```bash
cd /data/<user>/work/FrankenMotion
export FM_SETUP_DIR=/data/<user>/cache/lgfm_setup
export FM_ENV=/data/<user>/envs/frankenmotion
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/setup_school_environment.sh

$FM_ENV/bin/python -c 'import torch, hydra, pytorch_lightning; print(torch.__version__, torch.cuda.is_available())'
$FM_ENV/bin/python -m scripts.train_directional_fusion_ablation --help
```

脚本会在首次运行时创建 Python 3.10 环境并安装 CUDA 11.8 对应的 PyTorch 2.5.1；如果 `$FM_ENV/bin/python` 已存在，会复用该环境。不要在同一个环境中并发运行安装命令。若服务器已有经验证的项目环境，可直接令 `FM_ENV` 指向它，并只执行最后两条检查命令。

## 4. 启动 B1/B2/B3

先选择一个**已经完成的无动态图主干 run**作为 `REFERENCE_RUN`。该目录必须含有 `motion_stats/` 和 `text_stats/`。驱动只复制这两份归一化统计量，明确清空 `ckpt`、`init_ckpt`、`resume_dir`，因此不会把参照模型当作预训练模型。

```bash
cd /data/<user>/work/FrankenMotion
export FM_ENV=/data/<user>/envs/frankenmotion
export REFERENCE_RUN=/data/<user>/runs/lgfm_no_graph_3k
export OUTPUT_ROOT=/data/<user>/runs

nvidia-smi
```

确定某张卡可用后，分别启动。以下以物理 GPU 0 为例；`CUDA_VISIBLE_DEVICES=0` 后，训练进程内部看到的是 `cuda:0`。

```bash
# B1：无跨分支注意力
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$FM_ENV/bin/python" -u -m scripts.train_directional_fusion_ablation \
  --direction none \
  --reference-run "$REFERENCE_RUN" \
  --output-run "$OUTPUT_ROOT/ablation_b1_no_cross_attention_qk_b64_3k_v1"

# B2：仅 Global → Part
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$FM_ENV/bin/python" -u -m scripts.train_directional_fusion_ablation \
  --direction g2p \
  --reference-run "$REFERENCE_RUN" \
  --output-run "$OUTPUT_ROOT/ablation_b2_g2p_only_qk_b64_3k_v1"

# B3：仅 Part → Global
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$FM_ENV/bin/python" -u -m scripts.train_directional_fusion_ablation \
  --direction p2g \
  --reference-run "$REFERENCE_RUN" \
  --output-run "$OUTPUT_ROOT/ablation_b3_p2g_only_qk_b64_3k_v1"
```

正式实验使用 batch size 64、BF16、3000 epoch、seed 1234、`cross_attention_norm=qk_g2p`。配置快照可在以下文件审计：

```text
experiments/lgfm_3k/b1_no_cross_attention_config.json
experiments/lgfm_3k/b2_global_to_part_only_config.json
experiments/lgfm_3k/b3_part_to_global_only_config.json
```

训练程序拒绝覆盖已有 `--output-run`。需要后台执行时使用单独日志：

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$FM_ENV/bin/python" -u -m scripts.train_directional_fusion_ablation \
  --direction g2p --reference-run "$REFERENCE_RUN" \
  --output-run "$OUTPUT_ROOT/ablation_b2_g2p_only_qk_b64_3k_v1" \
  > logs/b2_g2p_3k.out 2>&1 &
tail -f logs/b2_g2p_3k.out
```

每个输出目录会保存 `planned_config.json` 和 `ablation_manifest.json`，用于核对开关、数据统计来源以及“fresh model and optimizer”约束。训练完成后确认 `logs/checkpoints/last.ckpt` 的 epoch 为 2999。

## 5. 统一评测

只在完整 3000 epoch 模型上做正式比较。所有模型必须使用相同 test split、相同 test prompt 列表、相同 20 次生成设置、同一 evaluator 和同一协议。建议固定如下生成参数：

```bash
export RUN_DIR="$OUTPUT_ROOT/ablation_b2_g2p_only_qk_b64_3k_v1"
CUDA_VISIBLE_DEVICES=0 "$FM_ENV/bin/python" -u eval_part.py \
  run_dir="$RUN_DIR" ckpt=last input_type=test num_runs=20 seed=9029 \
  guidance=1.0 device=cuda video_samples_per_run=0

export EXP_DIR=$(find "$RUN_DIR/generations/t2m" -maxdepth 1 -type d -name 'test_last_*' | head -n 1)
"$FM_ENV/bin/python" -u -m src.eval.run "$EXP_DIR" \
  --dataset_name=frankenstein-dataset --split=test --max_runs=20 \
  --protocol=guo --device=cpu --eval_model_dir=pretrained/eval_model \
  --base_splits_root=datasets/annotations --base_annotations_root=datasets/annotations
"$FM_ENV/bin/python" -u -m src.eval.run "$EXP_DIR" \
  --dataset_name=frankenstein-dataset --split=test --max_runs=20 \
  --protocol=guo+threshold --device=cpu --eval_model_dir=pretrained/eval_model \
  --base_splits_root=datasets/annotations --base_annotations_root=datasets/annotations
```

汇报时按 caption、action、parts-average 三个子集列出 FID+、MM-Dist、R@1/2/3、M2T、M2M、Diversity、Multimodality，以及 TemporalAccelGap、JerkGap、FootSlideGap、FootContactGap、TorsoLimbGap。置信区间应以测试样本为配对单位进行 bootstrap；20 次生成用于估计采样波动，不能被当成 20 个独立测试集。

## 6. 提交前检查

```bash
git status --short
git diff --check
```

仅提交模型、配置、脚本和文档。数据、`outputs_amass/`、检查点、`generations/`、缓存及日志都应保留在训练存储中。
