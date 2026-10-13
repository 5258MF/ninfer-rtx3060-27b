# Swift 1.5 IQ3XXS Portable Kit

**[简体中文](README.md) | English**

Version **0.1.4** shares the [Swift engine source](../engine/) with IQ2S. The standalone kit is available from the [shared downloads](../../README.en.md#downloads).

Extract the complete kit and run `启动.bat`. On first use, the launcher downloads `swift15_iq3xxs_mtp.ninfer`, verifies SHA256 and applies the bundled MTP-Q4 patch to produce `swift15_iq3xxs_mtpq4.ninfer`. No Python is required; allow about 21 GiB of free disk space. Existing converted models can be placed in `model/` or selected with `NINFER_IQ3_MODEL`. Runtime binaries and patch payloads are supplied in the cloud kit; this repository contains the launcher and full engine source.

Defaults: RK4V4, CUDA Graph off, CPU vision with automatic CPU threads, 128K logical context and `auto` resident capacity based on available VRAM. Press C to choose a recommendation or manual capacity. The engine allocates the complete instruction/tool prefix, retrieval and output per request; actual budgets appear in `[kvmem-alloc]` logs.

Use `启动.bat dryrun` or `预览参数.bat` to preview arguments and `测试.bat` after starting the server. API: `http://127.0.0.1:8084/v1`, model ID `qwen3.8-27b`. No test-specific DSH entry or session state is included. Custom builds can use `NINFER_IQ3_ENGINE` and `NINFER_IQ3_ENGINE_SHA256` to select the executable and its hash.

The 2026-10-13 checks covered short chat, large output requests, multiple tools, CPU images, edited history, resumed download with SHA256 verification and native PowerShell conversion of the full model. These checks do not qualify maximum context or continuous 32K/64K output. Final image features, language-model inference and KV still use the GPU.
