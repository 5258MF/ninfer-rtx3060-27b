# Swift 1.5 编译与开发

[源码入口](../swift15/README.md) · [使用指南](usage.md) · [技术实现](implementation.md)

## 从源码编译

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

环境变量：`NINFER_KVMEM_SINK_PAGES`（开头保留页数）、`NINFER_KVMEM_GEN_RESERVE_PAGES`（输出预留页数）、`NINFER_KVMEM_LONG_REUSE=1`。Windows 上开 KVMem 时 Host KV 自动用可分页内存（`NINFER_HOST_KV_PAGEABLE=0/1` 可以强制）。1 页 = 64 token。这些值由[启动器源码](../swift15/oneclick/launcher/launch.ps1)计算。

## 模型补丁与转换工具

懒人包不放模型，第一次启动时从魔搭下载 `swift15_iq2s_mtp.ninfer`（10,257,632,000 字节，SHA256 `2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e`），再用补丁转成 `swift15_iq2_s_mtpq4.ninfer`（10,155,600,640 字节，SHA256 `1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd`）。

自己生成模型补丁（在 `swift15/engine/` 里运行，要 `pip install gguf numpy torch`）：

```bash
python -X utf8 -m tools.mtpq4 swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer
python -X utf8 -m tools.mkpatch swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer patch_out
```

两个脚本已经放在引擎的 `tools/` 目录，会使用同一源码树中的 `tools.artifact`。

`mtpq4.py` 只改 MTP 的 7 个张量：注意力 query/gate（Q4）、key/value（Q8，引擎的 linear_pair 只收 Q8）、注意力 output（Q4）、MLP gate+up 和 down（Q4）；`input_projection` 没有对应形状的 Q4 内核，保留 Q6_K。量化用项目自带的 `quantize_matrix_mse`。其余对象逐字节复制。

## 升级现有包

重新编译后替换包内 `engine/ninfer-serve.exe`，保留完整依赖 DLL。已经下载和转换的模型可以继续使用。旧容量启动器补丁不包含首次显卡校准的引擎修复，具体适用范围见[故障排查](troubleshooting.md)。

源码和模型许可见[来源与许可](credits.md)。
