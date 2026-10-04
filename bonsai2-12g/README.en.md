# Bonsai2 12G Source and Launcher

**Language:** [简体中文](README.md) | English

The current source version is **0.1.2**, published as separate model-specific source releases with CPU retrieval, blocked scoring, and window allocation improvements. Cloud downloads still contain the previously uploaded **0.1.0** packages; fixed kits await upload. Existing download links have not been relabeled as 0.1.2. See the [changelog](../CHANGELOG.md).

This directory publishes the complete 12 GB engine source (formerly `ninfer-tree`), build script, and one-click launcher. It includes the sm_86, KVMem, rk8v4, and VRAM-budget changes, along with the three agent-compatibility fixes synchronized on 2026-10-04.

The current source version is **0.1.2** (see [`VERSION`](VERSION)). Bonsai2 12G has a [dedicated 0.1.2 source release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.2) with an asset containing only this build's source. The Baidu and Quark one-click packages are on the [project downloads page](../README.en.md#downloads).

## Directory

| Path | Contents |
|---|---|
| [`engine/`](engine/) | Complete CMake project, implementation, applications, tests, tools, and third-party source |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows/sm_86 build script; source and output paths are repository-relative |
| [`oneclick-8g/`](oneclick-8g/) | 8G launcher, stock settings, verification and manifest; shares `engine/` |
| [`oneclick/`](oneclick/) | Launcher, tests, public defaults, power settings, and verification scripts |
| [`engine/LICENSE`](engine/LICENSE) | Engine license; third-party folders keep their own licenses |
| [`engine/NOTICE`](engine/NOTICE) | Upstream notices |

## Use and Development

- Download, usage, parameters, test scope, and source build instructions: [Bonsai2 12G guide](../README-12GB.en.md).
- Recent fixes and release status: [CHANGELOG](../CHANGELOG.md).
- Swift 1.5 uses a separate engine source tree: [swift15](../swift15/README.en.md). Do not mix the engines or their model parameters.

The source does not include model weights, runtime DLLs, or build products. Building image support requires the FFmpeg development package; see the [12G build instructions](../README-12GB.en.md#source-and-build) for its location. Package verification scripts target a specific release build; update their hashes if you compile or repackage it yourself.

8G and 12G share the engine sources. Set `NINFER_RK4_SM86=ON` for the 8G rk4v4 engine; it uses a separate `build-rk4/` directory. See the [8G guide](../README-8GB.en.md). Prepare runtime DLLs and Q4 MTP `patch/lit.bin` from the kit; these are excluded from the source repository.
