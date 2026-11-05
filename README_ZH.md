# PDScope

[English](README.md) | [简体中文](README_ZH.md)

面向显存受限 MoE 推理的相位自适应跨层专家调度；论文已被 PPoPP 2027 接收。

## 论文

完整论文已随仓库提供：
[**PDScope: Breaking the Prefill-Decode Seesaw for Memory-Bound MoE Inference**](docs/pdscope-ppopp27.pdf)
（PPoPP 2027，2027 年 3 月 20–24 日，美国盐湖城）。

这是我们提交到 arXiv、在 ACM PPoPP 正式刊印之前的版本；正式版可能有些微小调整。等官方版本发布后，引用请以正式版为准，见[引用](#引用)。

## 概览

MoE 模型每个 token 只激活少量专家，但全部专家权重都必须可达。当专家放不进显存时，它们常驻主机内存，每一层都要逐个专家决定：在 CPU 上算，还是搬进 GPU 算。Prefill 与 decode 的需求恰好相反——prefill 希望尽可能多的 token 在 GPU 上算完，decode 希望尽量少的 PCIe 传输——因此任何单一静态调度策略必然在其中一相上吃亏。

PDScope 把“相位”作为一等调度输入，在相位切换时重新决定工作点：

- **AdaptSched：相位自适应跨层调度。** 用跨层成本模型把当前层需求与预测的下一层需求放在同一个 PCIe/CPU 预算下比较并放置专家；到层边界用真实路由替换预测后重新评估，预取深度随预测置信度调整。Prefill 走激进加载、深流水的运行点，decode 走保守 I/O、固定 CPU/GPU 配比的运行点。
- **零开销执行路径。** 常驻热集用 CUDA Graph 捕获；专家拷贝走零拷贝异步 I/O，并带 demand 暂存槽；decode 把整层 CPU 专家组交给一次原生 C++ 调用，替代约 20 个 Python/ATen 操作；fused 合并把逐专家 scatter 收敛成一次加权 index-add。
- **预测器。** 为了便于部署，本仓库提供的是因果 gate-fallback 预测器（用下一层的 gate 作用在当前激活上）。


## 关键结果

`Qwen3-30B-A3B`（BF16，48 层，128 专家，top-8），单张 A100 80 GB PCIe（约束到24GB以下），进程显存上限 24 GiB。单请求、生成 64 token、冻结的 ShareGPT 输入（取自论文的 latency 负载）：

| 输入 token 数 | prefill tok/s | decode tok/s | 端到端 tok/s | GPU 峰值 |
|---:|---:|---:|---:|---:|
| 512  | 343.92 | 7.984 | 6.821 | 19.98 GiB |
| 1024 | 597.18 | 7.689 | 6.460 | 20.12 GiB |
| 2048 | 1086.83 | 7.902 | 6.493 | 20.37 GiB |
| 4096 | 1671.54 | 7.765 | 6.058 | 20.83 GiB |

该配置下观测到的最优单请求 decode：**8.04 tok/s**。部分结果显著高于原论文披露结果，因为本仓库采用ktransformers提供内核。

## 快速开始

### 1. 环境要求

| 项目 | 参考机 |
|---|---|
| GPU | 1 × NVIDIA A100 80 GB PCIe，进程显存上限 24 GiB |
| CPU | 2 socket × 24 物理核（Intel Xeon Gold 5318Y，2 NUMA node） |
| 内存 | ≥ 256 GiB——仅 pinned BF16 专家主副本就约 58 GiB |
| 系统 / libc | 任意较新的 x86-64 Linux（实测 RHEL 9、glibc 2.34） |
| Python | 3.11，`torch 2.10.0+cu128`、`transformers 4.51.3`（见 `requirements.txt`） |
| 编译器 | g++ 11+ 与 Python 头文件；不需要 `nvcc` |
| 模型 | `Qwen3-30B-A3B` BF16 safetensors，需自行下载 |
| 数据集 | `ShareGPT_V3_unfiltered_cleaned_split.json` |

显存 ≥ 24 GiB 或核数不同的机器也能跑，但 decode 的 CPU/GPU 切分与工作线程数是主机属性，需要重新调参。

### 2. 安装

```bash
git clone https://github.com/nezha-Anonymous/PDScope.git pdscope
cd pdscope
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### 3. 编译 CPU 专家内核与扩展

```bash
git clone https://github.com/kvcache-ai/ktransformers.git "$HOME/kt-src"
git -C "$HOME/kt-src" checkout a02efe36b5802e7ca79a6b29146e01cf3b97d305
git -C "$HOME/kt-src" submodule update --init --recursive
export KT_SOURCE_ROOT="$HOME/kt-src"

export PYTHON_BIN="$(command -v python)"
./scripts/build_native.sh
```

全部用本机工具链编译到 `expertbench/native-build/`，不涉及系统升级。`WITH_AVX512=1 ./scripts/build_native.sh` 会额外编译 AVX-512 变体；默认仍是 AVX2，因为该内核每个 decode 层要流式读取约 66 MB，受 DRAM 带宽限制。

### 4. 指定模型与数据集

```bash
export PDSCOPE_MODEL=/path/to/Qwen3-30B-A3B
export PDSCOPE_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 5. 运行

```bash
./scripts/run_reference.sh               # -> runs/<UTC 时间戳>/result.json
./scripts/run_reference.sh results/run1  # -> results/run1/result.json
```

四个长度合计约 5 分钟（含模型加载）。每次运行会写入 `result.json`（吞吐、显存、完整配置）、`manifest.json`（主机、依赖版本、源码快照）、`prompt_sources.json`（冻结的请求）以及 `traces/<case>.json.gz`（逐层：放置决策、预取账本、主机侧时间窗口）。

想改 `--lengths` 时可以用等价的直接调用：

```bash
python benchmarks/pdscope_compare.py \
  --lengths 512 1024 2048 4096 --repetitions 2 \
  --pdscope-cpu-threads 18 --pdscope-decode-cpu-targets 6
```

实测成本表、热专家表与编译好的内核都会自动定位，执行路径默认就是实测参考配置；`--output` 默认写到 `runs/<时间戳>/result.json`。

### 6. 值得了解的开关

| 开关 | 作用 |
|---|---|
| `--pdscope-cpu-threads N` | 每个 NUMA subpool 的工作线程数（默认 18，共 36）。**最重要的旋钮**：把全部物理核交给内核会把控制线程挤到 SMT 兄弟核上，主机调用开销放大约 4 倍，decode 掉约 20 %。 |
| `--pdscope-decode-cpu-targets T` | decode 保留 T 个活跃专家在 CPU（参考机 6）。可以一次给多个值在同一会话内对比。请按自己的机器扫。 |
| `--cache-slots S` | 每层常驻专家数（默认 40 = 1920 槽，16.9 GiB）。真正的限制是显存上限。 |
| `--speculative-slots S` | 静态热集之外预取可占用的缓存槽数（默认 128）。 |
| `--pdscope-demand-slots S` | demand 暂存深度，即同时在途的专家拷贝数（默认 4）。 |
| `--attn-implementation` | attention 后端（默认 `sdpa`：省下 eager 的 softmax 工作区，4096 输入能用 40 槽正是靠它）。 |
| `--no-pdscope-graphs`、`--no-pdscope-native-dispatch`、`--no-pdscope-input-pregather`、`--no-pdscope-graph-output-pool`、`--no-torch-expandable-segments` | 关掉某个默认优化，便于与自己的系统做对照。 |

其余参数见 `--help`。

### 7. 不加载整模型检查构建（可选）

```bash
python benchmarks/pdscope_compare.py --mode two_layer
```

真实跑第 0/1 层（合成激活）：路由与独立 one-hot 下发一致、native vs FP32 相对 L2 ≤ 0.02、DMA 字节闭合、槽位生命周期与缓存句柄释放。

## 说明与边界

- **预测器。** 用因果 gate-fallback 预测器替代论文中可学习的相位感知预测器；每条 trace 都记录 `predictor_prior_hit_rate=0`。
- **范围。** 单请求、单模型（`Qwen3-30B-A3B`）、单卡。多 batch 服务与论文中的其它模型不在本仓库范围内。
- **对主机敏感。** 工作线程数与无关负载能让吞吐变化几十个百分点；请在空闲机器上测量并在比较前重新调参，测量协议见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

## 引用

PDScope 发表于 2027 年 ACM 并行编程原理与实践研讨会（ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming，PPoPP），会议于 2027 年 3 月 20 日至 24 日在美国盐湖城召开。

```bibtex
@inproceedings{pdscope2027,
  title     = {PDScope: Breaking the Prefill-Decode Seesaw for Memory-Bound MoE Inference},
  author    = {Yu, Enda and Dong, Dezun and Zhang, Zhaoning and Yang, Xiaojian and Yang, Weiling and Ma, Shaonan and Li, Dongsheng and Wu, Yongwei and Liao, Xiangke},
  booktitle = {Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '27)},
  year      = {2027}
}
```

## 许可

Apache-2.0（见 `LICENSE`）。ktransformers CPU 专家内核按固定 commit 拉取并本地编译，本仓库不重新分发该内核。

[ktransformers]: https://github.com/kvcache-ai/ktransformers
