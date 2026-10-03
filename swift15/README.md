# Swift 1.5 版完整源码、启动器、工具

懒人包和效果说明见上一级的 [README.md](../README.md)。最新引擎的完整源码在 [`engine/`](engine/)，Apache-2.0。KVMem、SM 自适应和启动容量修复已经合入；克隆本仓库后即可编译。

## 目录

| 路径 | 内容 |
|---|---|
| `engine/` | 完整的最新引擎源码，包含 CMake 工程、头文件、实现、应用、测试和第三方源码 |
| `scripts/build-sm86.bat` | Windows 编译脚本（sm_86，开 `NINFER_SLIM_3060`） |
| `engine/tools/mtpq4.py` | 把 Swift 1.5 的 MTP 草稿头从 Q6_K 压成 Q4（k/v 用 Q8） |
| `engine/tools/mkpatch.py` | 生成懒人包的模型补丁（`ops.txt` + `lit.bin`） |
| `oneclick/` | 懒人包的启动器（`launcher/launch.ps1`）、测试脚本、`设置.ini`、`使用说明.txt` |
| `hotfix/swift15-hotfix-kv-capacity.zip` | 旧包的启动器小补丁（约 20 KB），解压到原包目录覆盖即可 |

## 编译

### 1. 下载源码

```bash
git clone https://github.com/5258MF/ninfer-rtx3060-27b.git
cd ninfer-rtx3060-27b
```

也可以在 GitHub 上点击 Code → Download ZIP。引擎源码位于 `swift15/engine/`，上游基线是 Ryan-gsq 分支 **b06908b**，此前的改动均已合入。

### 2. 环境（Windows）

- CUDA 13.x（要带 cuBLAS；我们用的是 13.3）
- MSVC 2022（C++ 桌面开发）、CMake、Ninja
- FFmpeg 开发包（看图用）、zlib、libcurl（我们用 conda 里的）

### 3. 编译

改好 `swift15/scripts/build-sm86.bat` 开头的 CUDA、MSVC 和依赖路径，然后在仓库根目录运行：

```bat
swift15\scripts\build-sm86.bat
```

脚本默认使用仓库里的 `swift15/engine/`，输出到 `swift15/build/`；源码和构建路径使用纯英文路径。产物是 `swift15/build/apps/ninfer-serve.exe`。把它和运行要用的 DLL（cudart、cublas、cublasLt、FFmpeg、zlib、libcurl、libssh2、libcrypto、zstd 等）放到懒人包的 `engine\` 目录。

- 只编译 sm_86。RTX 30 系能跑，RTX 40 系（sm_89）二进制兼容也能跑；RTX 50 系不行，请用上游的预编译包。
- SM 数在运行时自动识别，不同型号不用重新编译。
- 改动头文件以后，完整重新编译可能要近 1 小时。

### 4. 必须加的启动参数

单卡 12 GB 上，`--device-state-slots 0 --prefill-chunk 512` 必须加，否则显存不够。完整参数由启动器生成，大致是：

```
ninfer-serve swift15_iq2_s_mtpq4.ninfer --model-id qwen3.8-27b
  --kv-dtype rk4v4 --gdn-state-fp16 --spec mtp --draft-tokens 3
  --max-context 204800 --kv-capacity auto --kvmem-window-pages <按空闲显存算>
  --host-kv-mib 8192 --device-state-slots 0 --prefill-chunk 512
  --cuda-graph-allowance-mib 144 --default-max-tokens <输出>
  --max-private-continuations 4 --embedding-host
  [--lm-head-draft] [--vision --vision-residency overlay --vision-max-merged 4096]
```

环境变量：`NINFER_KVMEM_SINK_PAGES`（开头保留页数）、`NINFER_KVMEM_GEN_RESERVE_PAGES`（输出预留页数）、`NINFER_KVMEM_LONG_REUSE=1`。Windows 上开 KVMem 时 Host KV 自动用可分页内存（`NINFER_HOST_KV_PAGEABLE=0/1` 可以强制）。1 页 = 64 token。怎么算这些值见 `oneclick/launcher/launch.ps1`。

## 启动容量修复（2026-10-03）

部分显卡启动旧引擎时会报 `Main KV page count is outside the target capacity curve`。读入分段会按实际 SM 数对齐，例如请求 `--prefill-chunk 512`，实际可能变成 640；旧代码仍按 512 计算 KV 页数，分配结果就低于规划器的容量下限。

修复已直接合入 [`engine/src/runtime/engine/model_instance.cpp`](engine/src/runtime/engine/model_instance.cpp)：使用规划器容量曲线的最低页数，保留 SM 自适应和读入分段对齐。KVMem 的总上下文存放在内存，显存只保留窗口；`--max-context 204800` 大于 `--kvmem-window-pages 1152` 对应的显存窗口是正常配置，无需为这条报错删除窗口参数。

- 最新完整包和下载链接见 [主 README](../README.md)。修复版 `ninfer-serve.exe` 的 MD5：`5998263BECD56D84A0A2954D7BE37CBF`。
- 旧包可用 [启动器小补丁](hotfix/swift15-hotfix-kv-capacity.zip)：关闭模型窗口，解压到原包目录，覆盖 `launcher\launch.ps1`。启动器只在遇到上述容量错误时关闭分段对齐并重试一次。
- 已在 RTX 3060 12GB 上验证默认 200K 配置启动和接口生成，并通过调整分段复现、修复容量越界；旧引擎配合新启动器也验证了自动重试。报错的 RTX 3060 Laptop 尚待用户复测。

## 已合入源码的改动

| 改动 | 内容 |
|---|---|
| KVMem | 移植 qzshch/ninfer-kvmem 的空洞块表、设备放置事务、稀疏工作集、滚动预填窗口、解码环形窗口和检索机制 |
| 3060 精简 | `NINFER_SLIM_3060` 保留单卡、单路、H24 注意力和 int8 / rk8v4 / rk4v4 内核 |
| 内存和显存 | Windows Host KV 用可分页内存；词嵌入放锁页内存；开头和输出按环境变量预留 |
| kv9 检索 | 按消息分段查询、对准换行、排除开头/保护块/最近块、CPU 打分矩阵化多线程；默认查询行数 1024 |
| 长对话复用 | 修复工作集恢复、长追问的 KV 覆盖范围和接续点问题 |
| 注意力和换入 | 跳过空洞页；页面换入按碎片程度自动选批量 gather |
| SM 自适应 | GDN 路线表运行时按真实 SM 数选择；rope 容量 = 6 × SM |
| 启动容量 | KVMem 自动容量使用规划器曲线下限，修复实际读入分段经 SM 对齐后变大造成的启动容量越界 |

## 模型补丁（懒人包里的 `patch\`）

懒人包不放模型，第一次启动时从魔搭下载 `swift15_iq2s_mtp.ninfer`（10,257,632,000 字节，SHA256 `2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e`），再用补丁转成 `swift15_iq2_s_mtpq4.ninfer`（10,155,600,640 字节，SHA256 `1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd`）。

自己生成模型补丁（在 `swift15/engine/` 里运行，要 `pip install gguf numpy torch`）：

```bash
python -X utf8 -m tools.mtpq4 swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer
python -X utf8 -m tools.mkpatch swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer patch_out
```

两个脚本已经放在引擎的 `tools/` 目录，会使用同一源码树中的 `tools.artifact`。

`mtpq4.py` 只改 MTP 的 7 个张量：注意力 query/gate（Q4）、key/value（Q8，引擎的 linear_pair 只收 Q8）、注意力 output（Q4）、MLP gate+up 和 down（Q4）；`input_projection` 没有对应形状的 Q4 内核，保留 Q6_K。量化用项目自带的 `quantize_matrix_mse`。其余对象逐字节复制。

## 致谢和许可

- 本目录代码：Apache-2.0（见 `LICENSE`），和上游一致。
- 上游引擎：[Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco)（Apache-2.0），它是 iamwavecut/ninfer-all 的分支，最早来自 Neroued/ninfer。
- KVMem 新引擎实现：[qzshch/ninfer-kvmem](https://github.com/qzshch/ninfer-kvmem) 的 `feature/kvmem` 分支（Apache-2.0）。相关实现已合入；源码保留原许可证和贡献者信息。
- KVMem 方法：kvmem-llama.cpp 项目和它的论文（arXiv 2609.04852）。
- 模型：Swift 1.5 Qwen3.8-27B，UkisAI（Swift Open License v1.0），基座 Qwen3.8-27B（Apache-2.0）。补丁只改 MTP 草稿头，详见上面。
