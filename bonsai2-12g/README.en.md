# Bonsai2 12G Source and Launcher

**Language:** [简体中文](README.md) | English

This directory publishes the complete 12 GB engine source (formerly `ninfer-tree`), build script, and one-click launcher. It includes the sm_86, KVMem, rk8v4, and VRAM-budget changes, along with the three agent-compatibility fixes synchronized on 2026-10-04.

The current source version is **0.1.0** (see [`VERSION`](VERSION)), corresponding to the Git tag [`v0.1.0`](https://github.com/5258MF/ninfer-rtx3060-27b/tree/v0.1.0). The Baidu and Quark packages are available from the [project downloads](../README.en.md#downloads).

## Directory

| Path | Contents |
|---|---|
| [`engine/`](engine/) | Complete CMake project, implementation, applications, tests, tools, and third-party source |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows/sm_86 build script; source and output paths are repository-relative |
| [`oneclick/`](oneclick/) | Launcher, tests, public defaults, power settings, and verification scripts |
| [`engine/LICENSE`](engine/LICENSE) | Engine license; third-party folders keep their own licenses |
| [`engine/NOTICE`](engine/NOTICE) | Upstream notices |

## Use and Development

- Download, usage, parameters, test scope, and source build instructions: [Bonsai2 12G guide](../README-12GB.en.md).
- Recent fixes and release status: [CHANGELOG](../CHANGELOG.md).
- Swift 1.5 uses a separate engine source tree: [swift15](../swift15/README.en.md). Do not mix the engines or their model parameters.

The source does not include model weights, runtime DLLs, or build products. Building image support requires the FFmpeg development package; see the [12G build instructions](../README-12GB.en.md#source-and-build) for its location. Package verification scripts target a specific release build; update their hashes if you compile or repackage it yourself.
