# Swift 1.5 版源码：引擎补丁、启动器、工具

懒人包和效果说明见上一级的 [README.md](../README.md)。这里是懒人包用到的全部代码，Apache-2.0。

## 目录

| 路径 | 内容 |
|---|---|
| `patches/` | 29 个补丁，基于 Ryan-gsq 分支的 **b06908b**，用 `git am` 打上 |
| `scripts/build-sm86.bat` | Windows 编译脚本（sm_86，开 `NINFER_SLIM_3060`） |
| `tools/mtpq4.py` | 把 Swift 1.5 的 MTP 草稿头从 Q6_K 压成 Q4（k/v 用 Q8） |
| `tools/mkpatch.py` | 生成懒人包的模型补丁（`ops.txt` + `lit.bin`） |
| `oneclick/` | 懒人包的启动器（`launcher/launch.ps1`）、测试脚本、`设置.ini`、`使用说明.txt` |
| `hotfix/swift15-hotfix-kv-capacity.zip` | 旧包的启动器小补丁（约 20 KB），解压到原包目录覆盖即可 |

## 编译

### 1. 拿源码、打补丁

```bash
git clone https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco.git ninfer-src
cd ninfer-src
git checkout b06908b
git am /path/to/swift15/patches/*.patch
```

29 个补丁能干净地打上（已在全新源码上验证），不需要手动解决冲突。

### 2. 环境（Windows）

- CUDA 13.x（要带 cuBLAS；我们用的是 13.3）
- MSVC 2022（C++ 桌面开发）、CMake、Ninja
- FFmpeg 开发包（看图用）、zlib、libcurl（我们用 conda 里的）

### 3. 编译

改好 `scripts/build-sm86.bat` 开头的路径，然后：

```bat
build-sm86.bat
```

产物是 `ninfer-serve.exe`。把它和运行要用的 DLL（cudart、cublas、cublasLt、FFmpeg、zlib、libcurl、libssh2、libcrypto、zstd等）放到懒人包的 `engine\` 目录。

- 只编译 sm_86。RTX 30 系能跑，RTX 40 系（sm_89）二进制兼容也能跑；RTX 50 系不行，请用上游的预编译包。
- SM 数在运行时自动识别（补丁 0028），不同型号不用重新编译。
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

补丁 0029 改为直接使用规划器容量曲线的最低页数，保留 SM 自适应和读入分段对齐。KVMem 的总上下文存放在内存，显存只保留窗口；`--max-context 204800` 大于 `--kvmem-window-pages 1152` 对应的显存窗口是正常配置，无需为这条报错删除窗口参数。

- 最新完整包和下载链接见 [主 README](../README.md)。修复版 `ninfer-serve.exe` 的 MD5：`5998263BECD56D84A0A2954D7BE37CBF`。
- 旧包可用 [启动器小补丁](hotfix/swift15-hotfix-kv-capacity.zip)：关闭模型窗口，解压到原包目录，覆盖 `launcher\launch.ps1`。启动器只在遇到上述容量错误时关闭分段对齐并重试一次。
- 已在 RTX 3060 12GB 上验证默认 200K 配置启动和接口生成，并通过调整分段复现、修复容量越界；旧引擎配合新启动器也验证了自动重试。报错的 RTX 3060 Laptop 尚待用户复测。

## 补丁说明

| 补丁 | 内容 |
|---|---|
| 0001–0009 | qzshch/ninfer-kvmem `feature/kvmem` 分支的提交（作者信息保留）：空洞块表、设备放置事务、内存端块检索索引、稀疏工作集、滚动预填窗口、解码时的环形窗口等 |
| 0010 | 该分支其余提交（95f2a71f..5299907）合并成一个补丁 |
| 0011 | 合并后的编译错误修正 |
| 0012 | KVMem 选项检查（只支持 int8 / rk8v4 / 单卡，后面 0018 加上 rk4v4） |
| 0013、0015 | `NINFER_SLIM_3060`：只保留单路 int8 / rk8v4 注意力内核，读入注意力只留 H24 |
| 0014 | Windows 上开 KVMem 时 Host KV 默认用可分页内存（`NINFER_HOST_KV_PAGEABLE` 可强制），锁页内存分不出 8 GB |
| 0016 | 旧式分配：开头保留、输出预留，用环境变量控制 |
| 0017 | 超窗长对话复用的修正：放宽工作集恢复检查等 |
| 0018 | 精简引擎里保留 rk4v4 内核，KVMem 支持 rk4v4 |
| 0019 | kv9 检索移植：按消息分段查询、对准换行、打分排除开头/保护块/最近块、CPU 打分矩阵化多线程 |
| 0020 | `--embedding-host`：词嵌入放锁页内存，GPU 零拷贝读，省约 360 MiB 显存 |
| 0021、0022 | 诊断用：`KV9_TIMING` 读入分段计时、`KV9_QMAX` / `KV9_QSHADOW` 挑块评测（默认关） |
| 0023 | 读入和小批量注意力跳过空洞页 |
| 0024 | 修复：超窗长对话里发一条超过约 100 token 的新消息时引擎崩溃（`KV coverage exceeds active entitlement`） |
| 0025 | 检索查询行数默认 1024（按评测选的） |
| 0026 | 页面换入时批量拷贝（gather，经映射锁页内存）；按碎片程度自动选：平均每段连续页少于 8 页才用（`KV9_GATHER_RUN`） |
| 0027 | 按 28 SM 调 GDN 门控路线表和 rope 整波容量（已被 0028 取代） |
| 0028 | SM 自适应：GDN 路线表恢复上游原版、运行时按真实 SM 数顺延；rope 容量 = 6 × SM |
| 0029 | KVMem 自动容量使用规划器曲线下限，修复实际读入分段经 SM 对齐后变大造成的启动容量越界 |

## 模型补丁（懒人包里的 `patch\`）

懒人包不放模型，第一次启动时从魔搭下载 `swift15_iq2s_mtp.ninfer`（10,257,632,000 字节，SHA256 `2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e`），再用补丁转成 `swift15_iq2_s_mtpq4.ninfer`（10,155,600,640 字节，SHA256 `1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd`）。

自己生成补丁（在打好补丁的源码根目录运行，要 `pip install gguf numpy torch`）：

```bash
python -X utf8 tools/mtpq4.py swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer
python -X utf8 tools/mkpatch.py swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer patch_out
```

（把 `tools/` 里的两个脚本复制到源码根目录再运行，它们要 import 源码里的 `tools.artifact`。）

`mtpq4.py` 只改 MTP 的 7 个张量：注意力 query/gate（Q4）、key/value（Q8，引擎的 linear_pair 只收 Q8）、注意力 output（Q4）、MLP gate+up 和 down（Q4）；`input_projection` 没有对应形状的 Q4 内核，保留 Q6_K。量化用项目自带的 `quantize_matrix_mse`。其余对象逐字节复制。

## 致谢和许可

- 本目录代码：Apache-2.0（见 `LICENSE`），和上游一致。
- 上游引擎：[Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco)（Apache-2.0），它是 iamwavecut/ninfer-all 的分支，最早来自 Neroued/ninfer。
- KVMem 新引擎实现：[qzshch/ninfer-kvmem](https://github.com/qzshch/ninfer-kvmem) 的 `feature/kvmem` 分支（Apache-2.0）。补丁 0001–0010 来自它，0002–0009 保留了原作者信息。
- KVMem 方法：kvmem-llama.cpp 项目和它的论文（arXiv 2609.04852）。
- 模型：Swift 1.5 Qwen3.8-27B，UkisAI（Swift Open License v1.0），基座 Qwen3.8-27B（Apache-2.0）。补丁只改 MTP 草稿头，详见上面。
