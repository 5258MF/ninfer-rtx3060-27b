# 2026-10-05 / 06 Scoring and Performance Records

**[简体中文](README.md) | English**

The main Chinese/English project READMEs show the current tables. This directory contains 19 original PPL reports and prompt-free timing data. K = 1024 tokens.

Reports 01/02 are IQ3XXS MTP Q4/Q6 artifacts; report 03 is the historical Bonsai2 quick run. Reports 04–19 are early custom-tiny/custom-wiki40k pipeline checks and context/stride sweeps. They are not comparable quick-mode quality scores. `ppl/catalog.json` records configurations, sample counts, original report names and SHA256. Original report bytes are preserved.

IQ3 scoring used the independent perplexity entry with an added host-embedding option: ninfer-ppl-1m-v1 quick mode, four streams, context4096/stride2048, rk4v4, GDN FP16, prefill/score tiles1024. Each artifact scored 261167 tokens in 124 windows; main-model PPL was 4.495637819904494 in both, with matching window scores at report precision. CausalScoring does not execute the MTP draft or vision tower; these results do not establish unchanged draft quality, acceptance rates or multimodal performance. The old IQ2S PPL row remains historical because its source report was not recovered here.

`performance/decode-samples.json` contains 10 configurations, one warmup and three measured requests for each text/code input. Text/code inputs were 56/49 tokens and generated 256 tokens. Original usage/timings and output/source hashes are retained; prompts and output text are omitted. Rates are engine decode medians, not end-to-end throughput. The tests used an RTX3060 12GB, Windows, full output head, MTP draft3, CPU vision and chat prefill512. Reaching the length limit does not certify complete or correct code.

`performance/prefill-audit.json` records exact2K/8K/16K inputs, one warmup and three measured requests per length, at most8 output tokens:90 measured samples and30 warmups. Timing/usage input counts agreed and cached tokens were zero. Prefill throughput and streaming TTFT are separate metrics. All prompts fitted in resident KV, so these rates do not cover long-history paging.

Graph pairs for the same model/KV used equal resident capacity; different models used different windows. IQ3 was tested at37K and Bonsai8 at48K, whereas current launcher defaults are IQ3 33K and Bonsai8 up to36K. These are historical benchmark windows, not new capacity guarantees. IQ3's first prefill attempt left223.68MiB and hit the old300MiB safety rule; the retry completed, and the exception remains in the audit. Small differences under sequential runs and changing temperature/desktop load do not prove a general speedup. Short-request checks do not qualify continuous32K/64K output or broad stability.

Full local evidence and original sessions remain in the user's local archive. No test-only DSH wrappers, prompts, logs or session state are published as part of the generic launcher.
