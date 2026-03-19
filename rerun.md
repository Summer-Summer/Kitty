# Kitty Rerun 实验指南

## 1. 分支状态总结

### 当前分支: `jisen/rerun`

基于 `main` + 1 个 commit (`561e21f`):

- **eval_helper.py bug fix**: 修复 HF baseline 模式下 `past_key_values` KeyError
- **新增 scripts/**: 完善的评估脚本（SLURM / 后台模式 / 消融实验 / HF 量化）
- **lm-evaluation-harness 子模块更新**

### main 最新 2 个 commit（已包含在 jisen/rerun 中）

| Commit | 作者 | 内容 | 影响 |
|--------|------|------|------|
| `ac9396e` Update utils_cli.py | Haojun | `channel_selection` 新增选项 `2` (Variance-based) | CLI 参数 |
| `5e99079` Update utils_quant.py | Haojun | 新增 Variance-based Channel Selection 策略 | 核心算法 |

**新增功能**: `--channel_selection 2` 使用方差（variance）而非幅度（magnitude）来选择需要提升精度的通道。

### `jisen_dev` 比 `jisen/rerun` 多的 2 个 commit — 不需要 port

| Commit | 内容 | 结论 |
|--------|------|------|
| `97a699f` temp storage | eval_helper bug fix + 旧版脚本 | bug fix 已 port，脚本已重写 |
| `55453bc` GPQA simple_eval | lm-eval 子模块引用变更 | jisen/rerun 已有更新的引用 |

---

## 2. 已完成的更名变更

`scripts/` 下 4 个脚本 + 2 个新增 rerun 脚本:

| 变更项 | 旧值 | 新值 |
|--------|------|------|
| CLI 命令 | `eval_rock_kv` | `eval_kitty` |
| CLI 参数 | `--eval_rock_kv` | `--eval_kitty` |
| 项目路径 | `$HOME/RoCK-KV/` | `$HOME/Kitty/` |
| 容器内工作目录 | `/workspace/RoCK-KV/eval_scripts` | `/workspace/Kitty/accuracy_simulation` |
| SLURM job name | `rock_kv_eval` | `kitty_eval` |

---

## 3. 环境准备 Checklist

- [x] git submodule 初始化 (`git submodule update --init --recursive`)
  - `third_party/transformers` @ `hf-4.53.2`
  - `third_party/lm-evaluation-harness` @ `kitty` 分支
- [x] Apptainer 容器构建 (`kitty_cuda121.def` → `/data/jisenli2/Kitty/build/kitty.sif`, 6.3G)
- [x] Overlay 创建 (`kitty.img`, 8G)
- [x] 在 overlay 中安装: transformers 4.53.2, lm_eval 0.4.9.1, lm-eval[math], kitty 1.0.0
- [x] 验证 `eval_kitty` → `/opt/conda/bin/eval_kitty` ✅
- [x] 验证 `eval_hf_kv` → `/opt/conda/bin/eval_hf_kv` ✅

### 容器构建命令

```bash
mkdir -p $HOME/Kitty/build && cd $HOME/Kitty/build
sudo apptainer build kitty.sif /data/jisenli2/Kitty/kitty_cuda121.def
apptainer overlay create --size 8192 kitty.img
```

### Overlay 安装

```bash
apptainer exec --nv --overlay $HOME/Kitty/build/kitty.img:rw $HOME/Kitty/build/kitty.sif bash -c "
    cd /workspace/Kitty
    pip install -e third_party/transformers
    pip install -e third_party/lm-evaluation-harness
    pip install -e .
"
```

---

## 4. Rerun 实验计划

### 4.1 目标

验证 **Variance-based Channel Selection** (`--channel_selection 2`) 在不同 promote_ratio 下的表现，与之前的 Magnitude-based (`--channel_selection 1`) 结果对比。

### 4.2 实验矩阵

**模型**: Qwen/Qwen3-8B
**Repeat**: 1 次（先跑通，后续可增加）
**Batch Size**: 32
**Max New Tokens**: 4096

**Tasks**:
- `gsm8k_cot_llama`
- `minerva_math_algebra`

**9 个 Config** (全部使用 `channel_selection=2`, `sink=32`, `kbits=2`, `vbits=2`, `promote_bit=4`):

| # | Config 名称 | promote_ratio | 含义 |
|---|-------------|---------------|------|
| 1 | sinkKIVI-K2V2 | 0.0 | 0% 通道提升，纯 K2V2 baseline |
| 2 | kChanBoost-12.5% | 0.125 | 12.5% 通道提升至 INT4 |
| 3 | kChanBoost-25% | 0.25 | 25% 通道提升至 INT4 |
| 4 | kChanBoost-37.5% | 0.375 | 37.5% 通道提升至 INT4 |
| 5 | kChanBoost-50% | 0.5 | 50% 通道提升至 INT4 |
| 6 | kChanBoost-62.5% | 0.625 | 62.5% 通道提升至 INT4 |
| 7 | kChanBoost-75% | 0.75 | 75% 通道提升至 INT4 |
| 8 | kChanBoost-87.5% | 0.875 | 87.5% 通道提升至 INT4 |
| 9 | sinkKIVI-K4V2 | 1.0 | 100% 通道提升，等效 K4V2 |

注: `promote_ratio=0.0` 和 `1.0` 不受 channel_selection 策略影响（全不选/全选），作为 sanity check。

**总计**: 9 configs × 2 tasks = **18 个实验**

### 4.3 脚本

| 脚本 | Task | 运行方式 |
|------|------|----------|
| `scripts/rerun_variance_gsm8k.sh` | gsm8k_cot_llama | 后台模式, 8 GPU 并行 |
| `scripts/rerun_variance_math.sh` | minerva_math_algebra | 后台模式, 8 GPU 并行 |

每个脚本跑 8 个 config（GPU 0-7），第 9 个 config (100% promo) 需要单独跑或等前面的完成后复用 GPU。但 0% 和 100% 的 channel_selection 策略无影响，可以直接用之前 Magnitude-based 的结果，所以实际只需新跑 7 个 config (12.5%~87.5%)。

### 4.4 运行命令

```bash
# 先跑 gsm8k
bash scripts/rerun_variance_gsm8k.sh
# 查看日志
tail -f ~/Kitty/log/rerun_var_gsm8k_*.log

# gsm8k 完成后跑 math
bash scripts/rerun_variance_math.sh
tail -f ~/Kitty/log/rerun_var_math_*.log
```

### 4.5 结果目录

```
eval_results_rerun_variance/
└── Qwen3-8B/
    ├── gsm8k_cot_llama/
    │   └── kitty_g128_b128_s32_sel2_k2_v2_pb4_pr<ratio>/
    │       ├── *_repeat_0.json
    │       └── *_summary.json
    └── minerva_math_algebra/
        └── ...

eval_logs_rerun_variance/
└── <node>/
    └── Qwen3_8B/
        ├── gsm8k_cot_llama/
        │   └── <gpu>_Qwen3_8B_gsm8k_cot_llama_s32_sel2_k2v2_pro<ratio>.log
        └── minerva_math_algebra/
            └── ...
```

### 4.6 与之前结果的对比

之前的实验使用 `--channel_selection 1` (Magnitude-based)，结果在 `eval_results_*` 目录中。
对比要点:
- 相同 promote_ratio 下，Variance vs Magnitude 的精度差异
- Variance-based 在哪些 promote_ratio 下表现更好/更差
- 两种策略的精度-效率 tradeoff 曲线

---

## 5. SIF/IMG 路径配置

构建完成后需确认脚本中的路径一致:

```bash
# scripts/rerun_variance_*.sh 中使用:
APPTAINER_SIF="$HOME/Kitty/build/kitty.sif"
APPTAINER_IMG="$HOME/Kitty/build/kitty.img"

# 如果沿用旧容器名:
# APPTAINER_SIF="$HOME/Kitty/build/kchanboost.sif"
# APPTAINER_IMG="$HOME/Kitty/build/kchanboost.img"
```

---

## 6. GPU 分配

当前脚本假设有 8 块 GPU (GPU 0-7)，每个 config 占 1 块 GPU。
如果 GPU 数量不同，修改 EXPERIMENTS 数组中的 GPU_ID 即可。
如果 GPU 不够，可以分批运行：先注释部分 config，跑完后换一批。

---

## 7. 容器版本冲突问题排查记录（2026-03-18）

### 问题根因

`kitty.sif` 基于 `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`，base image 自带：

| 包 | base image 版本 | third_party/transformers 要求 |
|----|----------------|-------------------------------|
| `tokenizers` | 0.22.2 | `>=0.21,<0.22` ❌ |
| `huggingface-hub` | 1.7.1 | `>=0.30.0,<1.0` ❌ |

脚本通过 `PYTHONPATH` 使用 `third_party/transformers`，它的 `dependency_versions_check.py` 在 import 时用 `importlib.metadata` 检查依赖版本。

由于 **Apptainer overlayfs 无法删除 base SIF 的 dist-info 文件**，即使在 overlay 中降级安装了正确版本，`importlib.metadata.version()` 仍会读到 base SIF 的旧 dist-info（如 `tokenizers-0.22.2.dist-info`），导致版本检查误报 ImportError。

### 已尝试但失败的方案

1. `pip install 'tokenizers>=0.21,<0.22'`（overlay :rw）— pip 显示已满足，但 dist-info 仍冲突
2. `pip install --force-reinstall 'tokenizers==0.21.4'`— pip show 返回 0.21.4，但 `importlib.metadata.version()` 仍返回 0.22.2
3. `pip install -e third_party/transformers`（editable install）— base SIF 的 `transformers/` 目录优先于 `.pth` 文件，加上 `huggingface-hub` 版本又冲突

### 最终解决方案：重建 SIF（kitty_v2.sif）

**关键教训**：`hqq` 的依赖链会拉入 `transformers-5.x + tokenizers-0.22.x`，覆盖 pin。**必须去掉 hqq**。

`kitty_cuda121.def` 的修改：
1. 删除 `pip install hqq`
2. 在所有 pip install 之后加入版本 pin：
   ```
   pip install 'tokenizers==0.21.4' 'huggingface-hub>=0.30.0,<1.0'
   ```

### 当前环境状态（已完成）

| 文件 | 路径 | 状态 |
|------|------|------|
| SIF | `/data/jisenli2/Kitty/build/kitty_v2.sif` | ✅ 正常 (6.3G) |
| Overlay | `/data/jisenli2/Kitty/build/kitty_v2.img` | ✅ 正常 (8G) |
| 旧 SIF | `/data/jisenli2/Kitty/build/kitty.sif` | ❌ 版本冲突，弃用 |
| 旧 Overlay | `/data/jisenli2/Kitty/build/kitty.img` | ❌ 弃用 |

Overlay 中已安装（editable）：
- `transformers 4.53.2`（`third_party/transformers` @ `hf-4.53.2`）
- `lm_eval 0.4.9.1`（`third_party/lm-evaluation-harness` @ `rock-kv`）
- `kitty 1.0.0`

### 从零搭建环境的完整步骤

如需在新机器上重建：

```bash
# 1. 构建 SIF（需要 sudo，约 30 分钟）
sudo apptainer build /data/jisenli2/Kitty/build/kitty_v2.sif /data/jisenli2/Kitty/kitty_cuda121.def

# 2. 创建 overlay
apptainer overlay create --size 8192 /data/jisenli2/Kitty/build/kitty_v2.img

# 3. 安装依赖（注意用 :rw）
apptainer exec --nv \
    --overlay /data/jisenli2/Kitty/build/kitty_v2.img:rw \
    /data/jisenli2/Kitty/build/kitty_v2.sif bash -c "
    pip install -e /data/jisenli2/Kitty/third_party/transformers
    pip install -e /data/jisenli2/Kitty/third_party/lm-evaluation-harness
    pip install -e /data/jisenli2/Kitty
"

# 4. 验证
apptainer exec --nv \
    --overlay /data/jisenli2/Kitty/build/kitty_v2.img:ro \
    /data/jisenli2/Kitty/build/kitty_v2.sif \
    bash -c "eval_kitty --help"
```

### 脚本配置注意事项

| 配置项 | 正确值 | 错误值（坑） |
|--------|--------|-------------|
| `APPTAINER_SIF` | `kitty_v2.sif` | `kitty.sif`（版本冲突） |
| `APPTAINER_IMG` | `kitty_v2.img` | `kitty.img`（版本冲突） |
| `HF_HOME` | `/data/shared/huggingface` | `/workspace/.cache/huggingface`（超 quota） |
| `BUFFER_LENGTH` | `128` | 未定义（运行报错） |
| `GROUP_SIZE` | `128` | 未定义（运行报错） |
| `PYTHONUNBUFFERED` | `1` | 未设置（日志输出延迟） |
