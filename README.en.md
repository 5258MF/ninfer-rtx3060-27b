# ninfer: Run 27B Models Locally on RTX 3060

**Language:** [简体中文](README.md) | English

Current sources: **Swift 1.5 shared IQ2S / IQ3XXS engine 0.1.5; Bonsai2 shared 8G / 12G source 0.1.6**. The available Swift one-click package remains IQ2S 0.1.2; the IQ3XXS package will be published later. All three current one-click packages use the same Baidu and Quark folder links below. See the [changelog](CHANGELOG.md). This main revision has not been released yet.

Run Swift 1.5 and Bonsai2 locally on Windows, with chat, image input, tool calling, and long-context KVMem. The one-click package starts after extraction; the model is downloaded on first launch.

## Choose a Build

| Build | GPU and VRAM | Model | Guide |
|---|---|---|---|
| **Swift 1.5 (recommended)** | RTX 30 series with 12 GB or more; RTX 40 series may work but has not been tested | Current IQ2_S package, about 9.55 GiB; shared source supports IQ2_S / IQ3_XXS | [Usage guide](#swift-15-setup-and-usage) |
| **Bonsai2 12G** | Tuned for RTX 3060 12 GB and intended for RTX 30 series | Swift-Bonsai-2 ternary model, about 7.7 GiB | [12G guide](README-12GB.en.md) |
| **Bonsai2 8G** | For RTX 30 series cards with 8 GB; tested with VRAM limited on a 12 GB card | Swift-Bonsai-2 ptq1, about 6.6 GiB | [8G guide](README-8GB.en.md) |

Swift 1.5 defaults to a 200K total context and can be configured up to 256K. KVMem selects some history for the active GPU window. See [test results and known limits](#test-results-and-known-limits) together with the context limits.

## Downloads

The one-click packages do not include model files. Extract the full archive and keep directories such as `engine`, `launcher`, and `patch`.

| Build | Archive | Baidu Netdisk | Code | Quark | Code |
|---|---|---|---|---|---|
| Swift 1.5 Q2S · 0.1.2 | `ninfer-3060-swift15-q2s-mtp-oneclick-0.1.2.zip` | [Baidu](https://pan.baidu.com/s/1Dyyb-SOULerfXGzDJqLlyQ) | `jqtg` | [Quark](https://pan.quark.cn/s/9fd59ead2458) | `YDvh` |
| Bonsai2 12G · 0.1.3 | `ninfer-3060-12g-bonsai2-oneclick-0.1.3.zip` | [Baidu](https://pan.baidu.com/s/1Dyyb-SOULerfXGzDJqLlyQ) | `jqtg` | [Quark](https://pan.quark.cn/s/9fd59ead2458) | `YDvh` |
| Bonsai2 8G · 0.1.3 | `ninfer-3060-8g-bonsai2-oneclick-0.1.3.zip` | [Baidu](https://pan.baidu.com/s/1Dyyb-SOULerfXGzDJqLlyQ) | `jqtg` | [Quark](https://pan.quark.cn/s/9fd59ead2458) | `YDvh` |

**Release status:** The three available one-click packages are in the same Baidu and Quark shared folders. Choose Swift 1.5 Q2S 0.1.2, Bonsai2 12G 0.1.3, or Bonsai2 8G 0.1.3 by archive name. The IQ3XXS engine source is shared with IQ2S, but its one-click package is not available yet. Build identification, fixes, and startup troubleshooting are in [CHANGELOG.md](CHANGELOG.md).

### Converted Abliterated NInfer Models

These Huihui Qwen3.8 27B Abliterated models have been converted to NInfer format and are available on Hugging Face:

- [Ternary Bonsai-2 NInfer](https://huggingface.co/fyb1214/Huihui-Qwen3.8-27B-Abliterated-Ternary-Bonsai-2-NInfer)
- [Swift 1.5 GSQ-RCO NInfer](https://huggingface.co/fyb1214/Huihui-Qwen3.8-27B-Abliterated-Swift-1.5-GSQ-RCO-NInfer)

Abliterated variants reduce refusal or safety filtering. Read each model card for provenance, licensing, and usage warnings before use. The Swift model card recommends research and controlled local inference, not public-facing services or use by minors.

Current main: Swift shared source 0.1.5 and Bonsai2 shared source 0.1.6; no Release yet. Historical source downloads: [Swift 0.1.3](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.3) and [Bonsai2 0.1.4](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.4), with one engine family per asset. Cloud-package versions are unchanged.

## Quick Start

The steps below use Swift 1.5 as an example:

1. Prepare Windows 10/11, an NVIDIA driver that supports CUDA 13, and at least 12 GB of VRAM. 32 GB of system RAM is recommended. Allow about 22 GB of free disk space for the first model download and conversion.
2. Download the one-click package and extract it to a short path, such as `D:\ninfer-swift15`.
3. Double-click `启动.bat`, review the settings, and press Enter. On the first run, the launcher downloads, verifies, and converts the model. The server is ready when the window shows `listening`.
4. Double-click `测试.bat` to check the response. To connect a client, use the endpoint and output limit shown in the generated `接入信息.txt`.

The default API base URL is `http://127.0.0.1:8084/v1` and the model ID is `qwen3.8-27b`. See the [Swift 1.5 usage guide](#swift-15-setup-and-usage) for settings, model preparation, and client setup. Common issues are covered [below](#common-issues); known startup errors are documented in [CHANGELOG](CHANGELOG.md#启动错误与处理).

## Page Guide

- [Swift 1.5 setup and usage](#swift-15-setup-and-usage): launch, settings, clients, and model preparation.
- [Test results and known limits](#test-results-and-known-limits), [implementation](#implementation), [source and build](#source-and-build), [common issues](#common-issues), and [credits and licenses](#credits-and-licenses).
- Full Bonsai2 guides: [12G](README-12GB.en.md) and [8G](README-8GB.en.md).
- [Changelog](CHANGELOG.md): release dates, build identification, fixes, and startup troubleshooting (Chinese).

## Swift 1.5 Setup and Usage

### Requirements

- 64-bit Windows 10/11. The target is an RTX 30 series GPU with at least 12 GB of VRAM. RTX 40 series may run but has not been tested.
- The NVIDIA driver must support CUDA 13; driver version 580 or later is recommended for the current package. A complete extraction does not require a CUDA or Python installation. VC runtime DLLs are included.
- 32 GB of RAM is recommended. Allow about 22 GB of free disk space for the first model setup.

Extract the complete package to a short path, such as `D:\ninfer-swift15`. Keep every DLL in `engine` beside `ninfer-serve.exe`.

### Start and Stop

1. Double-click `启动.bat`. The defaults are rk4v4, the full MTP output head, a 200K total context, and automatic output allocation.
2. Press Enter to confirm. On first use, confirm the model download and conversion when prompted. Downloads can resume.
3. Wait for `listening`. If this GPU has no calibration record yet, the engine also measures its execution routes, so the first start takes longer.
4. Run `测试.bat` to check the response and speed. Close the launcher window to stop the engine.

Run only one engine at a time. In the menu, Enter starts, C reconfigures, R rereads free VRAM, and Q exits.

```bat
启动.bat last
启动.bat dryrun
```

`last` starts with the saved settings. `dryrun` shows the arguments without starting the engine or downloading anything.

### KVMem and the Four Configurations

### KVMem allocation from the actual instruction prefix

**Current main source (2026-10-06):** Automatic KVMem mode sets only the resident budget and a finite API request ceiling that follows it. It does not preallocate fixed SYS, a fixed 36K/half split or a 32K output cap. The engine preserves the actual system/developer/tool prefix and assigns retrieval/output per request. Here `--default-max-tokens` also caps explicit requests; it is not merely a fallback for omitted fields. A legacy SYS/sink setting or bootstrap seed is not the actual prefix allocation. Client compaction policies remain client-owned; no DSH test launchers are bundled.

At startup, free VRAM determines total resident capacity C. Each request then tokenizes the complete rendered system/developer instructions and tool definitions as prefix S. Retrieval excludes S; output gets the remaining capacity.

| Total context | Recommended retrieval | Preferred target | Output reserve |
|---|---|---|---|
| 256K | 32K–36K | 36K | C − S − retrieval, at least 8K |
| 128K | 16K–18K | 18K | C − S − retrieval, at least 8K |
| 200K | 25K–28.125K | 28.125K | Same rule |

The full prefix is aligned to 64 tokens for Bonsai or 128 for Swift. KVMem no longer reserves a fixed 8K prefix or imposes a 16K/32K output cap. Under pressure, retrieval shrinks first, with a warning below the recommendation. A request is rejected if its full prefix and an 8K output reserve cannot fit. Actual generation still obeys the client's limit, remaining logical context and stopping conditions; an 8K reserve does not promise 8K of generated text.

Configure Harness before the first request: use total context for `contextWindow`, and the numeric `maxTokens` generated by the launcher. This is the request ceiling with a minimal prefix; a longer actual prefix automatically reduces this turn's output budget. Startup retries regenerate connection information; personal launchers synchronize dsh when enabled. Legacy manual OUT/ANSWER, SYS/SINK and WINDOW settings migrate to automatic allocation.

On the 12GB test card, the 8G configuration subtracts current desktop/driver occupancy before applying an 8GB total budget. A 20K test or low-free-memory capacity and a theoretical 67K capacity with the entire 8GB free are different scenarios. Small windows can run, but long-history retrieval can degrade.

The four supported combinations and their historical decode measurements:

| Configuration | Total context | Retrieval target | Output | Historical speed |
|---|---|---|---|---|
| rk4v4 + full | 200K recommended, up to 256K | 28.125K at 200K / 36K at 256K | Remaining capacity | about 43 tok/s |
| rk4v4 + lite | up to 128K | 18K | Remaining capacity | about 47 tok/s |
| rk8v4 + full | up to 128K | 18K | Remaining capacity | 14–22% slower historically |
| rk8v4 + lite | up to 128K | 18K | Remaining capacity | — |

- **rk4v4:** Both K and V cache values use 4-bit storage. On the 3060 it is 14–22% faster than rk8v4 because it reads less data in a memory-bandwidth-limited workload. Perplexity is only 0.10% higher.
- **Lite head:** MTP drafts are selected from 130,000 common tokens. It is about 10% faster with no measured change in hit rate, at the cost of about 340 MiB more VRAM and a smaller window.

### Change Settings

Press C after launching `启动.bat` to choose the KV format, MTP output head, vision, total context, and thinking limit; prefix and output are allocated automatically. Press Enter to keep the current value. The choices are saved to `设置.ini`.

You can also edit `设置.ini` in a text editor:

| Setting | Purpose |
|---|---|
| `KV` | rk4v4 / rk8v4 |
| `HEAD` | full / lite |
| `CTX` | Total context; default 204800 |
| `OUT` | auto: startup Harness ceiling, actual request budget calculated later |
| `SYS` | auto: complete tokenized system/developer/tool prefix |
| `THINK` | Thinking limit; 0 means unlimited |
| `POST_THINKING` | Switch sampling parameters for the answer after thinking; default 1, set 0 to disable |
| `POST_THINKING_TEMP` / `POST_THINKING_TOP_P` / `POST_THINKING_TOP_K` | Answer temperature, top-p, and top-k; blank uses engine defaults, temperature defaults to 0.2 |
| `POST_THINKING_SAMPLER` | Answer sampling preset, e.g. `temp=0.2,top_p=0.95` |
| `ADAPTIVE_MTP` | Adaptive MTP draft length; default 0 uses a fixed 3 tokens |
| `RECOVER_INVARIANT` | Try to recover supported internal invariant failures; default 1 |
| `VISION` | Enable image input |
| `HOST` / `PORT` | Listen address and port |
| `API_KEY` | Access key; blank by default |
| `MODEL_ID` | Model ID used by clients |

The window and automatic output vary with free VRAM. Use the values in the generated `接入信息.txt` after startup.

Inference controls are included in the current one-click packages. `RECOVER_INVARIANT` only handles internal errors for which recovery is implemented; it cannot recover from every error.

### Connect a Client

After `listening` appears, open the generated `接入信息.txt`. It contains the current settings and a DeepSeek Harness configuration example.

| Client field | Default or instruction |
|---|---|
| API type | OpenAI-compatible / Chat Completions |
| Base URL | `http://127.0.0.1:8084/v1` |
| Model ID | `qwen3.8-27b` |
| API key | If no key is configured, use `none` if the client requires a non-empty value |
| contextWindow | Match `CTX`; default 204800 |
| maxTokens | Use the value in the current `接入信息.txt` |

The server also supports Anthropic Messages and OpenAI Responses. Reasoning text is returned in `reasoning_content`. Send `enable_thinking: false` to disable reasoning. Supported reasoning levels are low, medium, and xhigh, plus reasoning disabled.

The server handles one request at a time; additional requests are queued. For LAN access, set `HOST` to `0.0.0.0` and configure `API_KEY`. The connection file lists the available address.

After changing dsh settings, fully exit the app and its tray process before reopening it; otherwise it may write its old in-memory settings back to disk.

In 0.1.0, large client output limits are capped by the server configuration. KVMem requests that branch from, edit, or compress earlier history fall back to a fresh prefill, which can be slower than continuing from the conversation end. With multiple tools, `required`/`any` is treated as `Auto` and does not guarantee a tool call. Endpoints that accept `strict:true` do not enforce strict JSON Schema constraints; Swift Responses still rejects `strict:true`. See the [compatibility changes and limits](CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开) (Chinese).

### Download and Convert the Model

Model source: [ModelScope Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer](https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer).

Consolidated cloud model library (Qwen 3.8 27B): [Baidu Netdisk](https://pan.baidu.com/s/1Dyyb-SOULerfXGzDJqLlyQ) (code: `jqtg`) · [Quark](https://pan.quark.cn/s/9fd59ead2458) (code: `YDvh`).

- Original download: `swift15_iq2s_mtp.ninfer`, 10,257,632,000 bytes.
- The launcher applies the package patch to create `swift15_iq2_s_mtpq4.ninfer`, 10,155,600,640 bytes.
- Only seven tensors in the MTP draft head are changed; all other model objects remain unchanged. SHA256 is checked after download and conversion.
- Conversion uses PowerShell; Python is not required. After the converted file passes verification, the launcher deletes the original download.
- You can download the original file yourself, put it in the `model` directory under its original name, and launch the conversion.

Hashes and instructions for creating the patch are in [Source and Build](#model-patch-and-conversion-tools). Existing converted models can be reused after an engine update.

### Package Contents

| Path | Contents |
|---|---|
| `启动.bat` / `测试.bat` | Launch and test entry points |
| `设置.ini` | User settings |
| `engine/` | Executable and required DLLs |
| `launcher/` | Launcher and test scripts |
| `patch/` | Data patch for model conversion |
| `model/` | Empty initially; downloaded and converted model |
| `版本信息.txt` | Build identification for the repaired package |
| `SHA256SUMS.txt` / `verify-kit-manifest.ps1` | Core-file hashes, DLL integrity, and Chinese filename checks |
| `verify-arch-engine.ps1` | Engine checks for short prompts, over-window controls, and multi-turn cache reuse |
| `logs/` / `接入信息.txt` | Runtime logs and generated client instructions |

`使用说明.txt` is available for offline use. See [test results and known limits](#test-results-and-known-limits) for speed and long-context recall scope.

The verification tools are included in the current source and one-click packages. After extracting the complete package, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\verify-kit-manifest.ps1
powershell -ExecutionPolicy Bypass -File .\verify-arch-engine.ps1
```

The first command checks package files. The second starts the test engine and uses VRAM; close other models first. By default it reads the model in the package's `model` directory; use `-ModelPath` to specify another location. `SHA256SUMS.txt` and the verification scripts match a specific release build. If you compile or modify the scripts and repackage, update the manifest and expected hashes in the scripts too.

## Test Results and Known Limits

These measurements and validation results are historical and are listed by build. Speeds from different tools, model formats, prompts, and output lengths are not directly comparable. This documentation update did not rerun the tests. A configurable 256K total context does not mean every part of history is recalled exactly: KVMem only places selected history in the active window on each turn.

### Quality (Perplexity; Lower Is Better)

IQ3XXS was scored on 2026-10-05 with the independent `ninfer-perplexity` scoring entry: `ninfer-ppl-1m-v1` quick mode, four streams, 4096 context / 2048 stride, rk4v4, GDN FP16 and host embeddings. Each artifact scored 261,167 tokens across 124 windows. PPL prefill/score tiles are 1024 tokens, distinct from chat prefill512. Bonsai2 and IQ2S columns below are historical references; the Bonsai2 raw report was recovered, but the IQ2S raw report was not found in this archive. Settings differ, so this table does not isolate weight-quantization effects.

| Domain | Swift-Bonsai-2 (historical int8) | IQ2S (historical) | IQ3XXS (rk4v4) |
|---|---:|---:|---:|
| Chinese reference | 7.89415 | 5.443 | 5.29369 |
| PG-19 | 8.65629 | 7.439 | 7.26160 |
| WikiText | 7.82674 | 6.321 | 6.21847 |
| NInfer code | 1.85972 | 1.716 | 1.69572 |
| Overall | 5.62843 | 4.587 | **4.49564** |

The IQ3XXS MTP Q6 and Q4 artifacts both scored **4.495637819904494**; all 124 windows matched at report precision. CausalScoring evaluates the main model and does not run the MTP draft or vision tower, so this does not establish unchanged draft quality, acceptance rate or multimodal behavior. [Q4 report](benchmarks/20261005/ppl/report-01.json), [Q6 report](benchmarks/20261005/ppl/report-02.json), [all 19 recovered reports](benchmarks/20261005/ppl/catalog.json). Early custom-corpus scores are pipeline/sweep records, not additional comparable quality certifications.

### Speed (RTX 3060 12 GB)

2026-10-05, RTX 3060 12GB / Windows; full output head, MTP draft3, CPU vision, prefill chunk512. Text/code inputs are 56/49 tokens, with 256 generated tokens; one warmup and three measured requests per case. Decode rates are medians in tokens/s, not end-to-end throughput.

| Configuration | Graph | Tested resident KV | Text decode | Code decode |
|---|---|---:|---:|---:|
| Swift IQ2S · rk4v4 | off | 73K | 42.21 | 44.93 |
| Swift IQ2S · rk4v4 | on | 73K | 43.24 | 46.12 |
| Swift IQ2S · rk8v4 | off | 49K | 42.19 | 47.13 |
| Swift IQ2S · rk8v4 | on | 49K | 43.27 | 48.33 |
| Swift IQ3XXS · rk4v4 | off | 37K | 44.34 | 46.03 |

Zero-cache prefill: exact 2K/8K/16K inputs, at most 8 output tokens; one warmup and three measured requests per input length. Rates below are prefill medians, with streaming TTFT shown separately. Prompt timing and usage counts agreed; cached tokens were zero.

| Configuration | Graph | 2K prefill | 8K prefill | 16K prefill | 16K TTFT (s) |
|---|---|---:|---:|---:|---:|
| Swift IQ2S · rk4v4 | off | 483.66 | 498.10 | 489.02 | 33.55 |
| Swift IQ2S · rk4v4 | on | 483.40 | 497.85 | 489.36 | 33.52 |
| Swift IQ2S · rk8v4 | off | 484.26 | 498.14 | 490.38 | 33.44 |
| Swift IQ2S · rk8v4 | on | 483.59 | 498.52 | 490.65 | 33.44 |
| Swift IQ3XXS · rk4v4 | off | 500.47 | 515.41 | 505.69 | 32.44 |

Ten configurations produced 90 measured prefill samples plus 30 warmups. Different models used different windows; Graph pairs for the same model/KV used the same window. Small differences are not a general speedup guarantee. All prefill inputs fit within resident KV; these rates do not cover long-history KV paging. The 37K IQ3 / 48K Bonsai8 test windows are historical: current launchers use IQ3 33K / Bonsai8 up to 36K. IQ3 first exceeded the old 300MiB safety rule (223.68MiB remaining), then completed a retry; both facts are retained. [Metrics and methods](benchmarks/20261005/README.md).

### Long Context

| Scenario | Result |
|---|---|
| 120K document with a hidden sentence | **Found**; follow-up questions and switching away from and back to the conversation also worked |
| 190K hidden sentence, tool calling, and multi-turn reuse | All five trials passed with 16K output. At 23K output, the smaller retrieval window caused the tool trial to fail |
| 250K document with a hidden sentence | **Not found**: the relevant chunk was not selected into VRAM; see known limits |
| Continue a 60K conversation after it exceeds the GPU window | 2–4 seconds per turn, down from about 140 seconds before the change |

### Test System

| Component | Configuration |
|---|---|
| GPU | RTX 3060 12 GB (28 SM, 360 GB/s memory bandwidth) |
| CPU | Ryzen 7 3700X |
| RAM | 32 GB (Host KV reserves 8 GB for 200K context and 10 GB for 256K) |
| OS | Windows 11 |
| Client | DeepSeek Harness desktop app; other OpenAI-compatible clients also work |

### Known Limits

- **The hidden sentence was not found in the 250K test.** Its chunk received a low retrieval score and was not selected into VRAM. For exact lookup, keep material around 120K–190K if possible. For very long input, send the material first and the question in a separate message; this improves recall.
- **The first prompt is slow:** about 9 minutes for 190K and 12 minutes for 250K. Follow-up turns are much faster.
- **One request at a time:** additional requests are queued.
- **Per-response output, including reasoning, is limited:** automatic allocation on the 3060 is about 23K–28K. Longer output is truncated; split large tasks into steps.
- **GPUs with more than 12 GB:** the active window has a cap, so extra VRAM is not currently used.
- **RTX 30/40 series only:** the engine is built for sm_86. For RTX 50 series, use Ryan-gsq's official prebuilt package.

### Test Method

- Speed uses the engine's `refbench`; perplexity uses `ninfer-perplexity`. Settings were alternated to reduce thermal-throttling effects.
- Long-context checks covered hidden-sentence retrieval, follow-up questions, tool calling, and switching between conversations; regressions were rerun after changes.
- The one-click package was tested for resumable download, integrity checks, patch conversion, startup on the target machine, and `测试.bat`.

## Implementation

The performance figures below are from the recorded tests. They do not imply that every update will reproduce the same gains. Release-specific fixes and status are in [CHANGELOG](CHANGELOG.md).

### 1. Run the New Engine on RTX 30 Series

Only the newer `ninfer-all` branch could read Swift 1.5's GGUF quantization blocks (including IQ2_S and IQ3_XXS), but its prebuilt binaries supported RTX 50 series only. The port:

- Builds for sm_86 and supplies dependencies such as cuBLAS.
- Uses `NINFER_SLIM_3060` to keep only the single-GPU, single-request attention kernels, reducing build time and memory use.
- Detects the SM count at runtime, so the same executable works on an RTX 3060 (28 SM) and larger GPUs without recompiling.

### 2. KVMem: 256K Context with 12 GB VRAM

- Based on qzshch/ninfer-kvmem's implementation adapted to the new engine: the full conversation stays in system memory (Host KV), while a sparse GPU page table preserves the original positions of pages that are not resident in VRAM.
- **Host KV uses pageable memory on Windows.** The previous pinned-memory approach could not allocate 8 GB reliably.
- Retrieval follows kv9: query by message, align selected chunks to line boundaries, exclude the fixed prefix, protected chunks, and recent content during scoring, and use multithreaded matrix operations on the CPU.
- Works with MTP speculative decoding and image input.

### 3. Keep Output Resident; Recompute Only New Content

- Reserve the full output allowance in VRAM so it is not evicted; always retain the fixed system-prompt prefix.
- Once history exceeds the window, later turns recompute only new content. In a 60K conversation, a follow-up turn dropped from about 140 seconds to **2–4 seconds**.
- The continuation point of a long conversation is retained when switching between conversations.

### 4. VRAM Savings

| Change | VRAM saved | Tradeoff |
|---|---:|---|
| Host token embeddings (`--embedding-host`) | about 360 MiB | None; speed unchanged |
| Vision weights in host memory (overlay) | about 280 MiB | First image takes about 1.7 seconds |
| Quantize MTP draft head to Q4 using a model patch | about 95 MiB | No measurable speed or hit-rate change |
| rk4v4 KV | about 260 MiB at 32K context | Perplexity +0.10% |

### 5. Speedups

- Attention skips pages that are not resident in VRAM.
- When pages move from host memory to the GPU, batched gather copies are selected based on fragmentation.
- The lite MTP output head is optional (+10.5%).

### 6. Launcher

- Calculates the window from free VRAM at startup and aims to leave about 150–200 MiB free. If the engine reports less actual capacity, the launcher reduces the window and retries.
- Output can be set to automatic and is allocated based on the window size.
- Shows the previous settings first: press Enter to start or C to walk through the recommended values and limits.

### Build Features in the Current Source

| Area | Details |
|---|---|
| KVMem | Sparse page table, device-placement transactions, sparse working set, rolling prefill window, decode ring window, and retrieval adapted from qzshch/ninfer-kvmem |
| 3060 slim build | `NINFER_SLIM_3060` keeps single-GPU, single-request, H24 attention, and int8 / rk8v4 / rk4v4 kernels |
| Host and device memory | Pageable Host KV on Windows; pinned host embeddings; environment-controlled prefix and output reservations |
| kv9 retrieval | Message-based queries, line-aligned chunks, excludes prefix/protected/recent content, multithreaded CPU scoring; default query rows 1024 |
| Attention and paging | Skips sparse pages; chooses batched gather based on fragmentation |
| SM adaptation | Selects GDN routes at runtime using detected SM count; rope capacity = 6 × SM |
| Windows VRAM budget | Reads free memory from CUDA and NVML, using the lower value when NVML is available; `NINFER_FREE_VRAM_MIB` can impose a lower budget |
| KVMem scoring | Supports single-segment queries and the path at the prompt-window boundary; log marker: `kvmem_score: SELECT` |
| Connection keepalive | Windows TCP keepalive 10s / 3s, HTTP keepalive 120s, read/write timeout 300s |

The last three items correspond to source changes synchronized on 2026-10-04. See the [release notes](CHANGELOG.md#2026-10-04swift-15-q2s-源码与启动器同步) for background; the benchmark tables retain the earlier measurements.

## Source and Build

The IQ2S and IQ3XXS builds share the full Swift 1.5 engine source in [`swift15/engine/`](swift15/engine/), including CMake files, implementation, apps, tests, and third-party source. The current IQ2S launcher and settings are in [`swift15/oneclick-iq2s/`](swift15/oneclick-iq2s/), with build scripts in [`swift15/scripts/`](swift15/scripts/). See the [Swift source index](swift15/README.en.md).

The Bonsai2 8G and 12G builds share the engine source and build script in [`bonsai2/`](bonsai2/README.en.md), with separate launchers and settings for each memory target. See the [12G build guide](README-12GB.en.md#source-and-build) and [8G guide](README-8GB.en.md).

### Build from Source

#### 1. Get the Source

```bash
git clone https://github.com/5258MF/ninfer-rtx3060-27b.git
cd ninfer-rtx3060-27b
```

Or select Code → Download ZIP on GitHub. Shared IQ2S / IQ3XXS engine source: `swift15/engine/`. Upstream baseline: Ryan-gsq branch commit **b06908b**; earlier changes are already integrated.

#### 2. Windows Requirements

- CUDA 13.x with cuBLAS (built with 13.3)
- MSVC 2022 C++ desktop workload, CMake, and Ninja
- FFmpeg development package for image input, plus zlib and libcurl (the recorded build used conda packages)

#### 3. Build

Set CUDA, MSVC, and dependency paths at the top of `swift15/scripts/build-sm86.bat`, then run from the repository root:

```bat
swift15\scripts\build-sm86.bat
```

The script uses `swift15/engine/` and writes to `swift15/build/`. Keep source and build paths ASCII-only. The output is `swift15/build/apps/ninfer-serve.exe`. Copy it and its runtime DLLs (including cudart, cublas, cublasLt, FFmpeg, zlib, libcurl, libssh2, libcrypto, and zstd) into the package's `engine\` directory.

- Builds for sm_86. RTX 30 series works; RTX 40 series (sm_89) may also run because of binary compatibility. RTX 50 series is unsupported; use the upstream prebuilt package.
- SM count is detected at runtime, so different models do not need separate builds.
- A full rebuild after header changes may take nearly an hour.

#### 4. Required Launch Arguments

On a single-GPU 12 GB system, `--device-state-slots 0 --prefill-chunk 512` are required to fit in VRAM. The launcher generates arguments similar to:

```text
ninfer-serve swift15_iq2_s_mtpq4.ninfer --model-id qwen3.8-27b
  --kv-dtype rk4v4 --gdn-state-fp16 --spec mtp --draft-tokens 3
  --max-context 204800 --kv-capacity auto --kvmem-window-pages <based on free VRAM>
  --host-kv-mib 8192 --device-state-slots 0 --prefill-chunk 512
  --cuda-graph-allowance-mib 144 --default-max-tokens <output>
  --max-private-continuations 4 --embedding-host
  [--lm-head-draft] [--vision --vision-residency overlay --vision-max-merged 4096]
```

Environment variables: `NINFER_KVMEM_SINK_PAGES` (fixed-prefix pages), `NINFER_KVMEM_GEN_RESERVE_PAGES` (output reservation pages), and `NINFER_KVMEM_LONG_REUSE=1`. On Windows, KVMem automatically uses pageable Host KV; `NINFER_HOST_KV_PAGEABLE=0/1` can override this. One page equals 64 tokens. The [launcher source](swift15/oneclick-iq2s/launcher/launch.ps1) calculates these values.

### Model Patch and Conversion Tools

The package does not include model weights. On first launch it downloads `swift15_iq2s_mtp.ninfer` from ModelScope (10,257,632,000 bytes; SHA256 `2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e`) and applies a patch to create `swift15_iq2_s_mtpq4.ninfer` (10,155,600,640 bytes; SHA256 `1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd`).

To create the patch yourself, run these commands in `swift15/engine/`. Install dependencies with `pip install gguf numpy torch`:

```bash
python -X utf8 -m tools.mtpq4 swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer
python -X utf8 -m tools.mkpatch swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer patch_out
```

Both scripts are in the engine's `tools/` directory and use `tools.artifact` from the same source tree.

`mtpq4.py` changes only seven MTP tensors: attention query/gate (Q4), key/value (Q8; the engine's linear_pair accepts Q8 only), attention output (Q4), MLP gate+up and down (Q4). `input_projection` has no matching Q4 kernel and stays Q6_K. Quantization uses the project’s `quantize_matrix_mse`. All other objects are copied byte-for-byte.

### Upgrade an Existing Package

After rebuilding, replace `engine/ninfer-serve.exe` and keep the full set of runtime DLLs. Existing downloaded and converted models can be reused. See [CHANGELOG](CHANGELOG.md) for the required fix versions.

## Common Issues

### GPU or Host Memory Allocation Failure

| Symptom | Action |
|---|---|
| VRAM allocation fails or the window does not fit | Close other GPU applications; press R to reread free VRAM or C to reduce output |
| The engine becomes very slow | Check for other GPU processes; release VRAM and restart |
| `cudaMallocHost` / `bad allocation` | Check system RAM and other model usage; reduce total context. A smaller GPU window does not fix every pinned-host-memory failure |
| Model already running or port in use | Stop the existing model or change `PORT` in `设置.ini` |

The launcher handles only capacity errors it recognizes. It cannot automatically recover from every allocation failure. For long inputs, also consider [long-context recall limits](#test-results-and-known-limits).

### Download, Conversion, and Client Issues

- Interrupted download: launch again; existing `.part` files will resume.
- Integrity check failed: confirm the model file is complete and download it again as described in the package.
- Patch missing: extract the full `patch` directory again. An already converted model can still be used.
- Client output was truncated: set the maximum output from the current `接入信息.txt`; reasoning also uses the output budget.
- dsh settings did not take effect: fully exit the app and tray process before reopening to avoid restoring old settings.
- Launcher window closes immediately: open CMD in the package directory and run `启动.bat` to see the error.
- DLL load error: extract the full package and keep runtime DLLs beside the engine; see [VC runtime and DLL troubleshooting](CHANGELOG.md#vc-运行库或其他-dll-加载错误) (Chinese).

Engine logs are in the package's `logs` directory. Remove API keys and other private information before sharing logs.

For missing DLLs, KV capacity-curve errors, or first-run GPU calibration failures, see [startup troubleshooting](CHANGELOG.md#启动错误与处理) (Chinese).

## Credits and Licenses

Source-code and model licenses are separate.

- **Swift 1.5 engine:** Apache-2.0 text in [swift15/LICENSE](swift15/LICENSE); upstream license and contributor notices remain in the source.
- **Swift-1.5 Qwen3.8-27B GSQ-RCO:** UkisAI (Swift Open License v1.0, attribution and redistribution allowed; commercial use by companies with annual revenue above USD 1 million requires a separate agreement with UkisAI); base model Qwen3.8-27B (Apache-2.0).
- **NInfer-format model:** converted and published by fyb423 on [ModelScope](https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer).
  - Changes required by section 4(b) of the Swift Open License: only seven MTP draft-head tensors are requantized from Q6_K to Q4 (query/gate, output, MLP) and Q8 (key/value). All other weights are byte-for-byte unchanged. The package ships only the patch; users download the model from ModelScope.
- **Engine:** Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco (Apache-2.0), a fork of iamwavecut/ninfer-all, originally from Neroued/ninfer.
- **KVMem:** qzshch/ninfer-kvmem (Apache-2.0), its adaptation to the new engine, and the kvmem-llama.cpp project and paper.
- Tuning notes published by the 3090 branch and other RTX 3060 users were also useful references.

For Bonsai-2 27B engine terms, see [engine/LICENSE](bonsai2/engine/LICENSE) and [NOTICE](bonsai2/engine/NOTICE). Third-party components retain their own licenses. Model and code licenses apply separately.

For 8 GB GPUs, see [Bonsai2 8G](README-8GB.en.md). For Swift 1.5, return to the [project home](README.en.md).
