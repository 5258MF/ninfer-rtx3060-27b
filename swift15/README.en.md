# Swift 1.5 Source and Launcher

**Language:** [简体中文](README.md) | English

This directory contains the complete Swift 1.5 engine source, Windows build script, launcher, and model conversion tools. The upstream baseline is Ryan-gsq commit `b06908b`. KVMem, runtime SM-count detection, and startup fixes are integrated.

The current source version is **0.1.0** (see [`VERSION`](VERSION)), corresponding to the Git tag [`v0.1.0`](https://github.com/5258MF/ninfer-rtx3060-27b/tree/v0.1.0). The Baidu and Quark 0.1.0 packages are available from the [project downloads](../README.en.md#downloads).

The latest Q2S source and launcher were synchronized on 2026-10-04. They include VRAM budgeting, content scoring, connection keepalive, inference controls, verification tools, and a multi-turn vision KV page-borrowing timing fix.

Three agent-compatibility changes were also synchronized that day: safe cache handling for mid-history branching and compaction, caps on output and reasoning budgets, and tool-parameter adaptation. Their implementation boundaries are listed in the [changelog](../CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开) (Chinese). The complete Bonsai2 12G source is in [bonsai2-12g](../bonsai2-12g/README.en.md).

## Directory

| Path | Contents |
|---|---|
| [`engine/`](engine/) | Complete engine: CMake project, implementation, applications, tests, and third-party source |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows build script for sm_86 |
| [`oneclick/`](oneclick/) | Launcher, test scripts, settings, and user guide |
| [`oneclick/verify-kit-manifest.ps1`](oneclick/verify-kit-manifest.ps1) | Integrity checks for release files and Chinese filenames |
| [`oneclick/verify-arch-engine.ps1`](oneclick/verify-arch-engine.ps1) | Engine checks for short prompts, over-window retrieval controls, and multi-turn reuse |
| [`engine/tools/mtpq4.py`](engine/tools/mtpq4.py) | MTP draft-head quantization utility |
| [`engine/tools/mkpatch.py`](engine/tools/mkpatch.py) | Builds model-conversion `ops.txt` / `lit.bin` patches |
| [`hotfix/swift15-hotfix-kv-capacity.zip`](hotfix/swift15-hotfix-kv-capacity.zip) | Launcher patch for an older package's capacity error; see the changelog for scope |

## Getting Started

- Download the package: [project downloads](../README.en.md#downloads).
- Configure and run it: [usage guide](../README.en.md#swift-15-setup-and-usage).
- Build from source: [build guide](../README.en.md#source-and-build).
- Learn about the internals: [implementation](../README.en.md#implementation).
- Troubleshooting: [startup fixes](../CHANGELOG.md#启动错误与处理) (Chinese).
- Release history: [CHANGELOG](../CHANGELOG.md).

Engine license: [LICENSE](LICENSE). Model sources and terms: [credits and licenses](../README.en.md#credits-and-licenses).
