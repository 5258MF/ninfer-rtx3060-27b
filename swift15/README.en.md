# Swift 1.5 Shared IQ2S / IQ3XXS Engine Source

**Language:** [简体中文](README.md) | English

The current engine source is **0.1.4** and is shared by IQ2_S and IQ3_XXS. The available Swift one-click package remains **IQ2S 0.1.2**; an IQ3XXS package has not been released yet and will get its own download entry later. Swift and Bonsai2 packages use the same Baidu and Quark links on the [project home](../README.en.md#downloads). See the [changelog](../CHANGELOG.md).

This directory has one Swift 1.5 engine source tree for both quantization formats. Their NInfer format mappings and GPU kernels are part of the same CMake project, so the engine source does not need to be duplicated. The IQ2S launcher is in `oneclick-iq2s/`; `oneclick-iq3xxs/` contains the IQ3XXS development profile. Both share `engine/`. The IQ3XXS profile is not a published portable package.

Current main source is **0.1.5** (`VERSION`). This revision has no Release yet; the problem releases were withdrawn. Historical source downloads: [Swift 0.1.3](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.3), [Bonsai2 0.1.4](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.4).

The 2026-10-05 source update adds CPU vision encoding with optional OpenBLAS acceleration, automatic thread calibration and caching, and a reference CPU fallback when OpenBLAS is unavailable. The same engine path serves IQ2S and IQ3XXS. The handoff records passing IQ2S/IQ3XXS image checks and CPU-backend unit tests; this does not qualify every image, CPU, or long-context configuration.

On 2026-10-04, the Swift engine also received VRAM budgeting, content scoring, connection keepalive, inference controls, and a multi-turn vision KV page-borrowing timing fix. Three agent-compatibility changes followed: safe cache handling for mid-history branching and compaction, output and reasoning-budget caps, and tool-parameter adaptation. See the [changelog](../CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开) (Chinese). The shared Bonsai2 8G / 12G source is in [bonsai2/](../bonsai2/README.en.md).

## Directory

| Path | Contents |
|---|---|
| [engine/](engine/) | Complete CMake project, implementation, applications, tests, and third-party source |
| [scripts/build-sm86.bat](scripts/build-sm86.bat) | Windows build script for sm_86 |
| [oneclick-iq2s/](oneclick-iq2s/) | Current IQ2S launcher, tests, settings, and user guide |
| [oneclick-iq2s/verify-kit-manifest.ps1](oneclick-iq2s/verify-kit-manifest.ps1) | Integrity checks for release files and Chinese filenames |
| [oneclick-iq2s/verify-arch-engine.ps1](oneclick-iq2s/verify-arch-engine.ps1) | Engine API, output-cap, tool round-trip, and vision checks |
| [engine/tools/mtpq4.py](engine/tools/mtpq4.py) | MTP draft-head quantization utility |
| [engine/tools/mkpatch.py](engine/tools/mkpatch.py) | Builds model-conversion ops.txt / lit.bin patches |
| [hotfix/swift15-hotfix-kv-capacity.zip](hotfix/swift15-hotfix-kv-capacity.zip) | Launcher patch for an older package capacity error; see the changelog |

## Getting Started

- The currently available package is IQ2S 0.1.2: [project downloads](../README.en.md#downloads).
- Configure and run it: [usage guide](../README.en.md#swift-15-setup-and-usage).
- Build from source: [build guide](../README.en.md#source-and-build).
- Learn about the internals: [implementation](../README.en.md#implementation).
- Troubleshooting: [startup fixes](../CHANGELOG.md#启动错误与处理) (Chinese).
- Release history: [CHANGELOG](../CHANGELOG.md).

Engine license: [LICENSE](LICENSE). Model sources and terms: [credits and licenses](../README.en.md#credits-and-licenses).
