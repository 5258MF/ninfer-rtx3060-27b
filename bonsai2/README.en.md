# Bonsai2 Shared 8G / 12G Source and Launchers

**Language:** [简体中文](README.md) | English

The current source version is **0.1.3**, published as separate model-specific source releases with CPU retrieval, blocked scoring, and window allocation improvements. Cloud downloads still contain the previously uploaded **0.1.0** packages; fixed kits await upload. Existing download links have not been relabeled as 0.1.3. See the [changelog](../CHANGELOG.md).

This directory contains the engine source (formerly `ninfer-tree`) and build script shared by the Bonsai2 8 GB and 12 GB builds, plus separate one-click launchers for each memory target. The builds use different model files and launch settings; the engine implementation, common fixes, and build entry point are shared. It includes the sm_86, KVMem, rk8v4, and VRAM-budget changes, along with the three agent-compatibility fixes synchronized on 2026-10-04.

The current source version is **0.1.3** (see [`VERSION`](VERSION)). The shared Bonsai2 8G / 12G source is available in the [0.1.3 release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.3) whose asset contains only the Bonsai2 source directory. The Baidu and Quark one-click packages are on the [project downloads page](../README.en.md#downloads).

## Directory

| Path | Contents |
|---|---|
| [`engine/`](engine/) | Complete CMake project, implementation, applications, tests, tools, and third-party source |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows/sm_86 build script; source and output paths are repository-relative |
| [`oneclick-8g/`](oneclick-8g/) | 8G launcher, stock settings, verification and manifest; shares `engine/` |
| [`oneclick/`](oneclick/) | 12G launcher, tests, public defaults, power settings, and verification scripts |
| [`engine/LICENSE`](engine/LICENSE) | Engine license; third-party folders keep their own licenses |
| [`engine/NOTICE`](engine/NOTICE) | Upstream notices |

## Use and Development

- Download, usage, parameters, test scope, and source build instructions: [Bonsai2 12G guide](../README-12GB.en.md).
- Recent fixes and release status: [CHANGELOG](../CHANGELOG.md).
- Swift 1.5 uses a separate engine source tree: [swift15](../swift15/README.en.md). Do not mix the engines or their model parameters.

The source does not include model weights, runtime DLLs, or build products. Building image support requires the FFmpeg development package; see the [12G build instructions](../README-12GB.en.md#source-and-build) for its location. Package verification scripts target a specific release build; update their hashes if you compile or repackage it yourself.

8G and 12G share the engine sources. Set `NINFER_RK4_SM86=ON` for the 8G rk4v4 engine; it uses a separate `build-rk4/` directory. See the [8G guide](../README-8GB.en.md). Prepare runtime DLLs and Q4 MTP `patch/lit.bin` from the kit; these are excluded from the source repository.
