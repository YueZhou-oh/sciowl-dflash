# SciOWL DFlash 推理包

单机 8 卡上的对照测试用的就是这里的脚本和 draft checkpoint。环境只作说明，不打进这个目录。

## 目录

- `scripts/serve_env.sh`：CANN、conda、NPU 环境变量。
- `scripts/serve_baseline.sh`：只起 SciOWL，不加载 DFlash。
- `scripts/serve_dflash.sh`：SciOWL + 本目录里的 DFlash draft。
- `scripts/run_tasks.sh`：8 类任务，48 并发，每条最多生成 3584 token。
- `scripts/bench_tasks.py`、`scripts/tasks_6k.py`：请求客户端和 8 条任务原文。
- `ckpt/dflash_epoch4/`：第 5 个 epoch 的 draft。训练目录里的 `epoch4_end` 和 `checkpoint_best` 都指向它。

目标模型不在这个包里。路径是

`/llm_infer_workspace/hedi/qwen-llama-eagle/SciOWL-Instruct-30B-A3B`

`ckpt/dflash_epoch4/` 里没有 `optimizer_state_dict.pt` 和 `scheduler_state_dict.pt`。vLLM 推理只读 `config.json`、`config.py` 和 `model.safetensors`。优化器状态约 1.2GB，留在原来的训练目录：

`output/dflash_sciowl-30b-a3b_8node_e5_offline_zlab_perfectblend_nothink_sq3072_n1420405_20261116_184722/checkpoints/4/`

## 这次测试的启动参数

- 1 台机器，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `--tensor-parallel-size 8`
- `--max-model-len 4096`
- `--max-num-batched-tokens 65536`
- `--max-num-seqs 64`
- `--gpu-memory-utilization 0.92`
- `--enforce-eager`
- `--no-enable-prefix-caching`
- DFlash：`--spec-method dflash --spec-tokens 15`
- 客户端：48 请求、并发 48、`max_tokens=3584`、`temperature=0`、`enable_thinking=false`
- 端口 `18180`

## 怎么跑

先起服务（二选一），等 `http://127.0.0.1:18180/health` 返回 200，再跑任务。客户端不要走代理。

```bash
bash sciowl_dflash_infer_pack/scripts/serve_baseline.sh
# 或
bash sciowl_dflash_infer_pack/scripts/serve_dflash.sh

bash sciowl_dflash_infer_pack/scripts/run_tasks.sh sciowl_dflash_infer_pack/task_results
```

启动只用本目录里的程序。`serve_dflash.sh` 直接调用 `python -m vllm.entrypoints.cli.main serve`，NPU 补丁在 `npu_adapter/`。不再依赖仓库里的 `scripts/launch_vllm.py`、`source` 或仓库外的 `npu_adapter`。

目标模型权重仍在机器上的 `TARGET_MODEL`（默认是 SciOWL 原路径）。Draft 权重在 `ckpt/dflash_epoch4/`。两边的 ckpt 都在之后，用下面两条命令即可。

## 环境依赖

下面是跑通上述服务时机器上的版本。不需要把 conda 或 CANN 打进本目录。

| 组件 | 版本或路径 |
|---|---|
| Python | 3.11.14，解释器 `/llm_infer_workspace/hedi/conda_unpack/dflash/bin/python` |
| conda 环境 | `/llm_infer_workspace/hedi/conda_unpack/dflash` |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post7 |
| vLLM | 0.30.0 |
| vllm_ascend | 0.30.0 |
| transformers | 5.19.0 |
| safetensors | 0.8.0 |
| CANN | 9.0.1，`/llm_infer_workspace/hedi/env_own/Ascend_9.0.1/cann/cann-9.0.1`（compiler `version.info` 的 timestamp `20260625_111312340`） |
| 驱动 | npu-smi 25.2.3 |
| NPU | 单机 8 × Ascend 910，每张 HBM 65536 MB |
| rtmeta stub | `/llm_infer_workspace/hedi/env_own/librtmeta_stub.so`，通过 `LD_PRELOAD` 加载 |
| 本目录补丁 | `npu_adapter/`（`sitecustomize.py` 在 `SPECFORGE_NPU_ENABLE=1` 时把 CUDA 调用转到 NPU）；`scripts/triton_host_cxx.sh` 作为 Triton 的 host 编译器 |

进程环境变量：

- `SPECFORGE_NPU_ENABLE=1`
- `TASK_QUEUE_ENABLE=1`
- `VLLM_KV_CACHE_LAYOUT=LBHNC`
- 不要设置 `VLLM_PORT`（vLLM 会把它当成内部端口基数）
- `all_proxy` / `http_proxy` / `https_proxy` 需要清掉，否则本机请求会走到 socks 代理

`source` 默认会带上 CANN 8.5.1，并且可能把别的 conda 环境放到 `PATH` 前面。`serve_env.sh` 在 source 之后改回 CANN 9.0.1，并用上面的 `dflash` 解释器启动，避免用到别的环境里的 `python` 或 `vllm`。
