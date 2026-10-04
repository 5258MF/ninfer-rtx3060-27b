@echo off
rem ============================================================
rem NInfer v1.0.8 - RTX 4090 (sm_89) - generic startup.
rem
rem Model file expected next to this script:
rem   qwen3_6_35b_a3b.ninfer   (see download_model.bat)
rem
rem If you have more than one GPU, set CUDA_VISIBLE_DEVICES to
rem the index of your GPU as seen by CUDA.
rem
rem Tune --max-context / --kv-capacity / --max-concurrency to
rem your VRAM and workload. With --wddm-evictable-budget the
rem engine budgets against TOTAL VRAM instead of the WDDM
rem process budget: on this 24 GB card a 260000-token
rem rk4v4-e8 pool at C=4 fits at 96% VRAM (see README.md).
rem ============================================================
set CUDA_VISIBLE_DEVICES=0
ninfer-serve.exe qwen3_6_35b_a3b.ninfer ^
 --host 127.0.0.1 --port 8080 ^
 --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 ^
 --wddm-evictable-budget --max-concurrency 2 --device-state-slots 2 ^
 --spec mtp --draft-tokens 3 --lm-head-draft ^
 --prefill-chunk 2048
pause
