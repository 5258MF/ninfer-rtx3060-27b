# Swift 1.5 IQ3XXS Development Launcher

**[简体中文](README.md) | English**

IQ3XXS shares [`../engine/`](../engine/) with IQ2S. This is a local development profile checked on an RTX 3060 12GB, not a published portable cloud package.

Defaults: rk4v4, CUDA Graph off, CPU vision with automatic CPU threads, 33K resident KV, 128K logical context, an API request ceiling that follows resident capacity, MTP-Q4/draft3 and the full output head. The engine allocates the complete instruction/tool prefix per request; see `[kvmem-alloc]` for actual retrieval and output budgets. Continuous 32K/64K output has not been qualified.

Place a built Swift engine and DLLs in `engine/`, and `swift15_iq3xxs_mtpq4.ninfer` in `model/`, or set `NINFER_IQ3_ENGINE` / `NINFER_IQ3_MODEL`. For your own engine build, set `NINFER_IQ3_ENGINE_SHA256` to its SHA256. The included `capacity-policy.json` records measurements on the test hardware, not a capacity guarantee. Override it with `NINFER_IQ3_POLICY` when using your own measured policy.

`启动.bat` opens the menu; `预览参数.bat` previews arguments. Only the verified rk4/noGraph/300MiB profile can start; other rows are preview-only. No model is downloaded and no existing inference process is terminated.

After starting the server, connect any compatible OpenAI/Anthropic client to `http://127.0.0.1:8084`. Model ID: `swift15-iq3xxs`; logical context: 131072. The API ceiling is finite and follows resident capacity; the engine sets actual output limits from each request’s full prefix and retrieval budget. No DSH test presets or launchers are bundled.

CPU vision still requires GPU memory for the language model, KV and final image features. Short tool chains passed in the handoff; the full Graph matrix, new-window stress and continuous 32K/64K output were not fully tested. See the [changelog](../../CHANGELOG.md).
