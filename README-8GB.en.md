# Bonsai2 8G

**Language:** [简体中文](README-8GB.md) | English

A Windows inference build for RTX 30 series GPUs with 8 GB VRAM. It uses the Swift-Bonsai-2 27B ptq1 model (about 6.6 GiB); the MTP draft head is converted to Q4.

**Testing was done on an RTX 3060 12 GB with engine VRAM capped; this has not been validated on an actual 8 GB GPU.** Real 8 GB performance and desktop VRAM headroom may differ. See [test scope and limits](#test-results-and-known-limits).

## Download and Start

Download `ninfer-3060-8g-oneclick.zip`. The cloud links and access code are on the [project home](README.en.md#downloads).

Requirements: Windows 10/11, an NVIDIA driver that supports CUDA 13, and 32 GB RAM recommended. Allow about 15 GB of free disk space for the first conversion. Extract the full archive and double-click `启动.bat`. The model will be downloaded and converted on first launch. Run `测试.bat` once the server shows `listening`.

## Page Guide

[Modes and Capacity](#modes-and-capacity) · [Model Setup and Settings](#model-setup-and-settings) · [Test Results and Known Limits](#test-results-and-known-limits) · [Implementation](#implementation) · [Common Issues](#common-issues) · [Credits and Licenses](#credits-and-licenses) · [Changelog](CHANGELOG.md)

## Launch and Connect a Client

Extract the complete package and run `启动.bat`. Press Enter to use the current settings or C to open the wizard. The first launch downloads the model; the 8G package also converts its MTP head. Run `测试.bat` after `listening` appears. Configure the client using the generated `接入信息.txt`.

Default Base URL: `http://127.0.0.1:8084/v1`. Model ID: `qwen3.8-27b`. The client context and output limits must match the launch settings.

## Modes and Capacity

As with the 12G build, four modes are available. The startup wizard lists recommended values and limits.

| Mode | Maximum context | Maximum response | Best for |
|---|---:|---:|---|
| **Standard** (recommended) | 56K (all on GPU) | 32K | Most reliable for everyday chat; the model sees the full context |
| **KVMem** (recommended; default) | **256K** | 16K | Long documents, long conversations, coding tool loops |
| Standard rk4v4 (not recommended) | 84K (all on GPU) | 32K | Above 56K when the model must see the entire context |
| KVMem + rk4v4 (not recommended) | **256K** | **32K** | Like KVMem, with up to 32K output; comparable to 12G KVMem + rk8v4 |

- Both KV formats use less VRAM than the int8 format in the 12G build:
  - **rk8v4:** K uses int8 and V uses 4-bit; used by the first two modes.
  - **rk4v4:** K and V both use 4-bit, allowing more context in the same VRAM. Tradeoffs: 10–15% slower generation, draft hit rate falling from 64% to 55%, and about 0.33% higher PPL. That is why it is marked “not recommended.”
- **KVMem GPU window:**

| Mode | KV on GPU | Retrieved history + output space |
|---|---:|---:|
| KVMem (default tier) | 44K | 28K + 16K |
| KVMem (48K tier) | 48K | 32K + 16K |
| KVMem + rk4v4 | 64K | 32K + 32K |

- **The 48K tier is opt-in.** It is suitable when the display is connected to the motherboard and rendered by an integrated GPU. It lets the engine use about 7.6 GiB (7760 MiB), leaving only about 430 MiB for the desktop; this may not be enough if the display is connected to the RTX card.

## Model Setup and Settings

The workflow is similar to the 12G package:

- The extracted package is about 0.9 GB; it does not include the model. On first launch it downloads `bonsai2_27b_swift_ptq1.ninfer` (6.6 GiB) from ModelScope, with direct access in China, supports resume, and verifies the download. You can also download it in a browser and put it in `model`.
- No development environment is needed. Install an NVIDIA driver version 580 or later. CUDA, Python, and the VC++ runtime do not need to be installed separately.
- Double-click `启动.bat`. Press Enter to reuse the previous settings or C to configure KVMem, KV format, VRAM tier (KVMem only), vision, context, output length, and reasoning length. Each step shows its recommended value and limit; out-of-range input is rejected.
- The launcher generates `接入信息.txt` with the API address, model ID, context window, maximum output, and a ready-to-use DeepSeek Harness example.
- Supports OpenAI, Anthropic Messages, and OpenAI Responses API formats, tool calling, and image input.
- Only Swift-Bonsai-2 (ptq1 + Q4 MTP) is included. The original Bonsai-2 option from the 12G package is not included.
- On a 12 GB GPU, the launcher suggests using the 12G build but does not block startup.

See [test results and known limits](#test-results-and-known-limits) and [implementation](#implementation) for details.

## Test Results and Known Limits

These are historical measurements, grouped by build. Speeds from different tools, model formats, prompts, and output lengths are not directly comparable. This documentation update did not rerun them. A configurable 256K context does not guarantee exact recall of all history: KVMem selects only part of the history for each active window.

### Speed

| Metric | Result |
|---|---:|
| Generation (`ninfer_bench`) | **about 45 tokens/s** (about 38 with rk4v4) |
| Real chat response (median) | **about 36–40 tokens/s** |
| Short prompts (`测试.bat` included in package) | 38–46 tokens/s |
| Long prompt processing (2048-token chunks) | about 620 tokens/s |
| Long generation when GPU throttles at 89°C | about 24 tokens/s |

- About 30% slower than the 12G build (`ninfer_bench`: 45 vs. 67). Nearly all of the difference comes from the ptq1 model itself; most existing speed optimizations target the 12G PQ2 format. The [8G VRAM-saving changes](#implementation) have little effect on speed.
- All measurements used an **RTX 3060 12 GB** with engine VRAM capped to stay at or below 7.5 GB. **No actual 8 GB GPU was tested.**

### Long Context

| Scenario | Result |
|---|---|
| Hidden sentence in a 260K-token document, with no clue in the question | Correct in both KVMem and KVMem + rk4v4 |
| Fill the output limit once (KVMem + rk4v4, 32K) | Generated all 32,768 tokens normally; engine VRAM did not increase |
| First prompt with 260K tokens | About 12.5–14 minutes (300–340 tokens/s, slower near the end) |
| 57K-token document fully resident on GPU (standard mode) | Found in both trials |
| 64K-token document (KVMem and standard rk4v4) | Normal |

### VRAM

Engine-only peak usage, measured with `nvidia-smi` during testing:

| Mode | Peak engine use |
|---|---:|
| Standard | 7628 MiB |
| KVMem (default 44K tier) | 7659 MiB |
| KVMem (48K tier) | 7760 MiB |
| Standard rk4v4 | 7627 MiB |
| KVMem + rk4v4 | 7617 MiB (also 7617 after a full 32K response) |

Every mode passed the same tests for text, vision, tool calling, multi-turn chat, and 8K hidden-sentence retrieval.

### Test System

| Component | Configuration |
|---|---|
| GPU | RTX 3060 8 GB target (28 SM); actual test used a 12 GB card limited to 7.5 GB engine VRAM |
| RAM | 32 GB recommended (KVMem 256K history uses about 7–8 GB, plus about 1 GB pinned for embeddings and vision); with 16 GB, use standard mode |
| Disk | About 14 GB temporary space during conversion; about 7.3 GB after conversion |
| OS | Windows 10/11 |
| Client | DeepSeek Harness desktop app; other OpenAI-compatible clients also work |

### Known Limits

- **Not tested on an actual 8 GB card.** All results were collected on a 12 GB RTX 3060 with engine use capped at 7.5 GB.
  - The 8 GB RTX 3060 has 240 GB/s memory bandwidth versus 360 GB/s on the 12 GB version, so generation is expected to be slower; the difference has not been measured.
  - If the display is connected to this GPU, Windows, browsers, and other desktop apps also use VRAM. With the engine at 7.5 GB, only about 0.5 GB remains. If startup fails or is very slow, reduce context one tier or close GPU-heavy apps such as games and browsers with hardware acceleration.
  - If the display can use integrated graphics, nearly all discrete VRAM remains available to the model; the KVMem 48K tier may also fit.
- **RTX 30 series only:** the package binary is compiled specifically for sm_86.
- **About 30% slower than the 12G build** because the ptq1 model is slower.
- **One request at a time:** additional requests are queued.
- **KVMem output, including reasoning, is limited to 16K** in standard KVMem. Longer output is truncated. Standard and KVMem + rk4v4 allow 32K.
- **KVMem sees only selected history on each turn.** The GPU window contains 28K of history (32K for KVMem + rk4v4), less than in the 12G build. Details may be missed in very long conversations. Use standard mode (up to 56K) when the model must see the whole context.
- **The first 260K prompt takes over ten minutes.** Follow-up turns are much faster.

### Test Method

- Speed uses the engine's `ninfer_bench`, alternating each change on and off over three rounds.
- VRAM was measured by recording usage before startup, running the full test set, and subtracting the initial use from peak usage. Limits were found tier by tier below 7680 MiB, with a 30 MiB margin at 7650 MiB.
- Each tier had to pass basic chat, image input, tool calling, multi-turn chat, and the engine's 8K hidden-sentence test. KVMem also had to pass a 260K hidden-sentence test. The built-in 256K sample includes the answer in the question, so a new hidden sentence was used; both values had to be returned without clues.
- The package was also tested across nine wizard paths, model conversion and verification, resumable downloads, each engine binary on the target system, and `测试.bat`.

## Implementation

Performance figures below are from the recorded tests and are not guaranteed for future updates. The 8G build retains the 12G port, performance work, bug fixes, KVMem, rk8v4, and host-resident vision weights; see the [12G implementation guide](README-12GB.en.md#implementation). These changes were added to fit the build within 8 GB.

### 1. Switch to the ptq1 Model

- The 12G build uses PQ2 (7.7 GiB). The 8G build uses the smaller **ptq1** format (about 1.6 bits, 6.6 GiB), reducing the model file by about 1.1 GiB and reducing GPU weight memory.
- The RTX 3060 8 GB has the same 28 SM as the 12 GB version, so a separate engine build is not needed.
- We did not compare answer quality between ptq1 and PQ2.

### 2. Quantize the MTP Draft Head to Q4

- **MTP must remain enabled.** Turning it off saves only 478 MiB but drops generation from 45 to 11–12 tokens/s. Without MTP, ptq1 uses an unoptimized one-token path; MTP can verify four tokens at once.
- The original MTP head uses 8-bit weights. Five matrices are requantized to 4-bit Q4, saving about **210 MiB**. Speed remains 45.2–45.6 tokens/s, and the draft hit rate rose slightly from 62.5% to 64.1%.
- MTP only proposes draft tokens; the main model verifies each token. Quantization error affects the hit rate, **not the response content**.
- Three engine changes enable Q4 MTP. The original model path is unchanged.

### 3. Keep Token Embeddings in Host Memory

- The embedding table (token ID to vector) is 265 MiB. Each generated token reads one row (about 1 KB), so the table stays in RAM and the GPU reads the needed row over PCIe.
- Measured speed was unchanged (45.3 vs. 45.2–46.4 tokens/s), with an identical draft hit rate.

### 4. Other VRAM Savings

- **Vision weights in host memory**, as in the 12G build: saves about 260 MiB and adds about 0.1 seconds per image.
- **Reduce prompt chunking from 1024 to 512:** saves 85 MiB of temporary buffers with little change in prompt speed.
- **Disable extra cache slots:** the package serves one request at a time, so they are not needed.

### 5. Enable KVMem with rk4v4

- The original engine disallowed KVMem with rk4v4. Source inspection showed the relevant code did not depend on KV format, so only that check was removed; compute code was unchanged.
- The GPU window can hold 64K with rk4v4 (44K with rk8v4), allowing up to 32K output and 32K history. The 260K hidden-sentence test passed, and a full 32K response completed normally.
- The rk8v4 and rk4v4 kernels have the same symbol names and cannot be linked into one executable, so the package contains both engine binaries in one folder. The launcher selects one based on the mode.

### 6. Convert the Model in the One-Click Package

- ModelScope provides the original ptq1 model with an 8-bit MTP head. Conversion with Python on a user's machine would require several dependencies, so the package uses a patch instead.
- The original and Q4 versions differ in only five MTP matrices. The patch records which regions to copy from the original and which to replace; the added data is 226 MB.
- Conversion uses plain PowerShell and takes seconds. The result is SHA256-checked and compared byte-for-byte with the expected output before the original file is deleted.

### VRAM Breakdown (ptq1 + Q4 MTP; Vision and Embeddings in Host Memory)

| Component | Size |
|---|---:|
| Weights | 5.47 GiB |
| Current conversation loop state (48 linear-attention layers) | about 147 MiB |
| Other temporary buffers | about 205 MiB |
| KV per 1024 tokens | rk8v4: 27.7 MiB; rk4v4: 18 MiB |
| Vision components remaining in VRAM | about 68 MiB |
| KVMem overhead | about 240 MiB |

VRAM use is already close to the practical minimum. Other ideas, such as host-resident output layers or 3-bit MTP, either cut speed roughly in half or save too little memory.

## Common Issues

- Startup failure or sudden slowdown: close other GPU applications, reduce context or output by one tier, then restart.
- Host-memory allocation failure: close other models and reduce total context. With 16 GB RAM, use standard mode.
- Interrupted download: launch again; the existing download will resume. If verification fails, follow the package instructions to redownload.
- Client output was truncated: use the maximum output in the current `接入信息.txt`; reasoning also counts against the output budget.
- dsh changes did not take effect: exit the app and tray process before reopening. To inspect a closing launcher, open CMD in the package folder and run `启动.bat`.
- DLL load error: extract the full package and keep engine DLLs in the same folder. See [VC runtime and DLL troubleshooting](CHANGELOG.md#vc-运行库或其他-dll-加载错误) (Chinese).

The Swift 1.5 startup fixes have their own scope; see [startup troubleshooting](CHANGELOG.md#启动错误与处理) (Chinese). Remove API keys and other private information before sharing logs.

## Credits and Licenses

- **Bonsai-2 27B ternary model:** prism-ml. **Swift-Bonsai-2**, including ptq1: fyb423 on ModelScope.
- **ninfer engine:** the original author's ternary inference engine, RTX 3060 upgrade package, and tuning notes.
- **KVMem:** kvmem-llama.cpp and its paper.
- **rk8v4 / rk4v4:** implementation from ninfer-all.
- Public tuning notes from the 3090 branch and other RTX 3060 users were also useful.

See [engine/LICENSE](bonsai2-12g/engine/LICENSE) and [NOTICE](bonsai2-12g/engine/NOTICE) for the engine terms and upstream notices. Third-party directories retain their own licenses; model and code licenses apply separately.

For 12 GB GPUs, see [Bonsai2 12G](README-12GB.en.md). For Swift 1.5, return to the [project home](README.en.md).
